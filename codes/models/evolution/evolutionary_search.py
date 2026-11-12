"""Small, offline evolutionary architecture search around a frozen WikiClick ranker.

Run from this directory: .venv/bin/python -m codes.models.evolution.evolutionary_search
No collection, test-time training, or changes to existing checkpoints.
"""
import argparse
import copy
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn

from codes.search.compare_search import load_graph
from codes.models.raw.ml_model import DistanceModel, pair_features
from codes.models.next_link.next_link_experiment import (DATA, Ranker, evaluate_rank, grouped_batch,
                                  navigate, ranking_loss, summarize)
from codes.api.sample_dataset import save_json

AXES = {"features": ["title", "semantic", "both"], "width": [16, 32, 64],
        "depth": [1, 2], "activation": ["relu", "gelu"]}
SPACE = [dict(zip(AXES, values)) for values in itertools.product(*AXES.values())]


def key(config):
    return json.dumps(config, sort_keys=True)


def mutate(parent, rng, seen):
    """One-gene mutation; use an unseen random immigrant if neighbors run out."""
    neighbors = [c for c in SPACE if key(c) not in seen and
                 sum(c[k] != parent[k] for k in AXES) == 1]
    available = neighbors or [c for c in SPACE if key(c) not in seen]
    if not available:
        raise ValueError("Architecture space exhausted")
    return copy.deepcopy(rng.choice(available))


class Refinement(nn.Module):
    """Preserve the trained base and learn a zero-initialized residual branch."""
    def __init__(self, base, config):
        super().__init__()
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.config = dict(config)
        dimensions = 1024 if config["features"] == "both" else 512
        layers = []
        for _ in range(config["depth"]):
            layers.extend([nn.Linear(dimensions, config["width"]),
                           nn.ReLU() if config["activation"] == "relu" else nn.GELU()])
            dimensions = config["width"]
        layers.append(nn.Linear(dimensions, 1))
        self.adapter = nn.Sequential(*layers)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, candidates, goals, title, semantic):
        inputs = []
        if self.config["features"] in ("title", "both"):
            inputs.append(pair_features(title[candidates], title[goals]))
        if self.config["features"] in ("semantic", "both"):
            inputs.append(pair_features(semantic[candidates], semantic[goals]))
        values = torch.cat(inputs, -1) if len(inputs) == 2 else inputs[0]
        return self.base(candidates, goals, title, semantic) + self.adapter(values).squeeze(-1)


def load_selected(path, graph_id):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["graph_id"] != graph_id:
        raise ValueError("Checkpoint and graph differ")
    base = Ranker(DistanceModel().state_dict(), semantic=True)
    model = Refinement(base, checkpoint["config"]) if checkpoint["config"] is not None else base
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def fitness(record):
    """Success first, then shortest-route count, loss, and model size."""
    return (record["validation_routes"]["successes"],
            record["validation_routes"]["optimal_routes"],
            -record["validation_rank"]["loss"], -record["parameters"])


def measure(model, val, queries, graph, index, title, semantic):
    return {"validation_rank": evaluate_rank(model, val, title, semantic),
            "validation_routes": summarize(navigate(graph, queries, index, model, title, semantic)),
            "parameters": sum(p.numel() for p in model.parameters())}


def fit(base, config, train, val, queries, graph, index, title, semantic, epochs, seed):
    # Identical genotype gets identical initialization/order in both search arms.
    candidate_seed = seed + int(hashlib.sha256(key(config).encode()).hexdigest()[:7], 16)
    torch.manual_seed(candidate_seed)
    model = Refinement(base, config)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=.0003)
    generator = torch.Generator().manual_seed(candidate_seed)
    best = measure(model, val, queries, graph, index, title, semantic)
    state, selected_epoch = copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "train_loss": None, **best}]
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        order = torch.randperm(len(train), generator=generator).tolist()
        for start in range(0, len(order), 32):
            batch = [train[i] for i in order[start:start + 32]]
            optimizer.zero_grad()
            logits = grouped_batch(batch, title, semantic, model)
            loss = torch.stack([ranking_loss(v, r["optimal"]) for r, v in zip(batch, logits)]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.adapter.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(batch)
        metrics = measure(model, val, queries, graph, index, title, semantic)
        history.append({"epoch": epoch, "train_loss": total / len(train), **metrics})
        if fitness(metrics) > fitness(best):
            best, state, selected_epoch = metrics, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(state)
    return model, {"config": config, "selected_epoch": selected_epoch,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "history": history, **best}


def query_sample(graph, goals, count, seed, excluded=()):
    """Distinct destinations; reachable, >=2-click queries outside the old suite."""
    if count > len(goals):
        raise ValueError("Need at least one held-out destination per query")
    rng, blocked, queries = random.Random(seed), set(excluded), []
    selected = rng.sample(sorted(goals), count)
    for goal in selected:
        distances = nx.single_source_shortest_path_length(graph.reverse(copy=False), goal)
        sources = sorted(n for n, d in distances.items() if d >= 2 and (n, goal) not in blocked)
        if not sources:
            raise ValueError(f"No unused reachable source for destination {goal}")
        source = rng.choice(sources)
        queries.append({"source": source, "target": goal, "optimal_clicks": distances[source]})
    return queries


def paired(a, b):
    assert [(r["source"], r["target"]) for r in a] == [(r["source"], r["target"]) for r in b]
    result = {}
    for name, value in [("success", lambda r: r["path"] is not None),
                        ("optimal_route", lambda r: r["path"] is not None and r["clicks"] == r["optimal_clicks"])]:
        differences = np.array([int(value(y)) - int(value(x)) for x, y in zip(a, b)])
        draws = np.random.default_rng(733).choice(differences, (5000, len(a)), replace=True).mean(1) * 100
        result[name] = {"gain_percentage_points": float(differences.mean() * 100),
                        "conditional_95_percent_interval": np.quantile(draws, [.025, .975]).tolist(),
                        "improved_queries": int((differences > 0).sum()),
                        "worsened_queries": int((differences < 0).sum())}
    return result


def run(args):
    if args.population < 2 or args.generations < 2 or args.population * args.generations > len(SPACE):
        raise ValueError("Use population >=2, generations >=2, and at most 36 candidates per arm")
    if args.epochs < 1 or not 1 <= args.samples <= 100:
        raise ValueError("Use positive epochs and 1..100 samples")
    # Refuse accidental overwrites, including completed runs.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    graph, graph_id = load_graph(DATA)
    manifest = json.loads((DATA / "manifest.json").read_text())
    splits = manifest["destination_splits"]
    assert all(not set(splits[a]) & set(splits[b]) for a, b in itertools.combinations(splits, 2))
    original = Path("models/next_link/ranking_1000")
    features = torch.load(original / "features.pt", map_location="cpu", weights_only=True)
    assert features["graph_id"] == graph_id
    title, semantic = features["title_vectors"], features["semantic_vectors"]
    index = {pageid: i for i, pageid in enumerate(features["page_ids"])}
    train = json.loads((DATA / "train_decisions.json").read_text())
    val = json.loads((DATA / "validation_decisions.json").read_text())
    for rows, split in [(train, "train"), (val, "validation")]:
        assert all(features["page_ids"][r["goal"]] in splits[split] for r in rows)
    checkpoint_path = original / "seed_207/semantic_rank.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["graph_id"] == graph_id
    base = Ranker(DistanceModel().state_dict(), semantic=True)
    base.load_state_dict(checkpoint["state_dict"])
    val_queries = query_sample(graph, splits["validation"], args.samples, 1201)
    save_json(args.output / "validation_queries.json", val_queries)
    baseline = measure(base, val, val_queries, graph, index, title, semantic)
    records, winners = {}, {"baseline": base}
    baseline_record = {"id": "baseline", "config": None, "selected_epoch": 0, **baseline}
    for arm in ("evolution", "random"):
        rng, seen, population, evaluated = random.Random(args.seed), set(), [], []
        best_record, best_model = baseline_record, base
        for generation in range(args.generations):
            children = []
            for member in range(args.population):
                parent = None
                if arm == "evolution" and population:
                    parent = max(rng.sample(population, min(2, len(population))), key=fitness)
                    config = mutate(parent["config"], rng, seen)
                else:
                    config = copy.deepcopy(rng.choice([c for c in SPACE if key(c) not in seen]))
                seen.add(key(config))
                model, record = fit(base, config, train, val, val_queries, graph, index,
                                    title, semantic, args.epochs, args.seed)
                record.update(id=f"{arm}_{len(evaluated):02d}", generation=generation,
                              parent=parent["id"] if parent else None)
                evaluated.append(record)
                children.append(record)
                if fitness(record) > fitness(best_record):
                    best_record, best_model = record, model
                save_json(args.output / f"{arm}_candidates.json", evaluated)
                save_json(args.output / "status.json", {"state": "searching", "arm": arm,
                          "completed_candidates": len(evaluated), "generation": generation})
                print(f"{record['id']} {config} epoch={record['selected_epoch']} "
                      f"val success={record['validation_routes']['successes']}/{args.samples} "
                      f"optimal={record['validation_routes']['optimal_routes']}/{args.samples}", flush=True)
            population = sorted(population + children, key=fitness, reverse=True)[:args.population]
        records[arm] = {"selected": best_record, "candidates": evaluated}
        winners[arm] = best_model
        torch.save({"state_dict": best_model.state_dict(), "config": best_record["config"],
                    "graph_id": graph_id, "base_checkpoint": str(checkpoint_path),
                    "selected_id": best_record["id"]}, args.output / f"{arm}_selected.pt")
    # Selection is finished. Test labels and journeys are generated only now.
    old_queries = json.loads((original / "evaluation_queries.json").read_text())
    test_queries = query_sample(graph, splits["test"], args.samples, 1202,
                                [(r["source"], r["target"]) for r in old_queries])
    save_json(args.output / "test_queries.json", test_queries)
    routes, test = {}, {}
    for name, model in winners.items():
        navigate(graph, val_queries[:5], index, model, title, semantic)  # warm inference
        rows = navigate(graph, test_queries, index, model, title, semantic)
        for row in rows:
            path = row["visited_path"]
            assert all(graph.has_edge(a, b) for a, b in zip(path, path[1:]))
            assert len(path) == len(set(path))
            if row["path"] is not None:
                assert path[-1] == row["target"] and row["clicks"] >= row["optimal_clicks"]
        routes[name], test[name] = rows, summarize(rows)
        save_json(args.output / f"{name}.routes.json", rows)
    result = {"graph_id": graph_id, "nodes": len(graph), "edges": graph.number_of_edges(),
              "seed": args.seed, "samples": args.samples, "training_decisions": len(train),
              "validation_decisions": len(val), "population": args.population,
              "generations": args.generations, "epochs_per_candidate": args.epochs,
              "baseline": baseline_record, "search": records, "test": test,
              "paired_vs_baseline": {name: paired(routes["baseline"], routes[name]) for name in records},
              "elapsed_seconds": time.perf_counter() - started,
              "protocol": "Frozen seed-207 semantic ranker; zero-output residual adapters trained on existing train decisions. "
              "Tournament parent selection, one-gene mutation, elitist survivor selection; no parent weight inheritance. "
              "Random arm has identical candidate/epoch budgets and genotype-dependent training seeds. "
              "Epoch and architecture selection lexicographically maximize validation route success, optimal routes, "
              "negative ranking loss, negative parameter count. Baseline is eligible for rollback. "
              "20-click navigation, no revisits, shared direct-goal check. One query per test destination. "
              "Test source/destination pairs exclude the previous suite, but test destinations were previously evaluated. "
              "One graph and one search seed; intervals condition on these models. No live requests or test-time updates."}
    save_json(args.output / "results.json", result)
    save_json(args.output / "status.json", {"state": "complete", "elapsed_seconds": result["elapsed_seconds"]})
    print(json.dumps({"test": test, "paired": result["paired_vs_baseline"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/evolution/evolution_100"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1203)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
