"""Reproducible 1,000-article ranking and training-error correction experiment.

Stages: prepare (existing links), fetch (public Wikipedia intros/categories), run.
No test routes or labels are used to mine mistakes or select checkpoints.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import networkx as nx
import numpy as np
import torch
from torch import nn
from .scoring import Ranker, grouped_batch, ranking_loss
from ..heuristic.correction import mistake_weights

from wikiclick_core.API.wiki_api import USER_AGENT, query_api
from wikiclick_core.model.navigator.offline_search import load_graph, search
from wikiclick_core.model.title_feature.sha256 import fast_title_features
from wikiclick_core.model.neural.network import DistanceModel, pair_features
from wikiclick_core.API.training_data import save_json

DATA = Path("data/graphs/ranking_1000")
OUT = Path("models/next_link/ranking_1000")


def prepare():
    source = json.loads(Path("data/graphs/sample_1100/graph.json").read_text())
    ids = sorted(n["pageid"] for n in source["nodes"])
    keep = set(random.Random(611).sample(ids, 1000))
    content = {"nodes": [n for n in source["nodes"] if n["pageid"] in keep],
               "edges": [e for e in source["edges"] if e["source"] in keep and e["target"] in keep]}
    graph_id = "sha256:"+hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    DATA.mkdir(parents=True, exist_ok=True)
    frozen = {"schema_version": 1, "directed": True, "graph_id": graph_id, **content}
    if (DATA/"graph.json").exists():
        assert json.loads((DATA/"graph.json").read_text()) == frozen
    else:
        save_json(DATA/"graph.json", frozen)
    graph, _ = load_graph(DATA)
    goals = sorted(graph)
    random.Random(612).shuffle(goals)
    splits = {"train": goals[:800], "validation": goals[800:900], "test": goals[900:]}
    save_json(DATA/"manifest.json", {"graph_id": graph_id, "source_graph_id": source["graph_id"],
        "nodes": len(graph), "edges": graph.number_of_edges(), "destination_splits": splits,
        "scope": "Deterministic 1000-node induced subset of previously collected 1100; no new link collection. Text features are collected separately and have a later timestamp."})
    print(f"Prepared {len(graph)} articles, {graph.number_of_edges()} directed links", flush=True)


def fetch():
    frozen = json.loads((DATA/"graph.json").read_text())
    cache = DATA/"semantic_sources"; cache.mkdir(exist_ok=True)
    ids = sorted(n["pageid"] for n in frozen["nodes"])
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        pending = [i for i in ids if not (cache/f"{i}.json").exists()]
        for offset in range(0, len(pending), 20):
            batch = pending[offset:offset+20]
            params = {"pageids": "|".join(map(str, batch)), "prop": "extracts|categories|info",
                      "exintro": 1, "explaintext": 1, "exlimit": 20,
                      "cllimit": "max", "clshow": "!hidden"}
            records = {}
            continuation = {}
            while True:
                response = query_api(client, {**params, **continuation})
                for page in response.get("query", {}).get("pages", []):
                    pageid = page.get("pageid")
                    if pageid not in batch:
                        continue
                    record = records.setdefault(pageid, {"pageid": pageid, "categories": [],
                        "collected_at": datetime.now(timezone.utc).isoformat(), "source": "https://en.wikipedia.org/w/api.php"})
                    record["title"] = page.get("title", "")
                    if "extract" in page:
                        record["intro"] = page["extract"]
                    if "lastrevid" in page:
                        record["lastrevid"] = page["lastrevid"]
                    record["missing"] = bool(page.get("missing", False))
                    record["categories"].extend(c["title"] for c in page.get("categories", []))
                if "continue" not in response:
                    break
                continuation = response["continue"]
                time.sleep(.15)
            for pageid in batch:
                record = records.get(pageid, {"pageid": pageid, "missing": True, "intro": "", "categories": []})
                record["categories"] = sorted(set(record["categories"]))
                save_json(cache/f"{pageid}.json", record)
            print(f"Semantic sources cached: {len(ids)-len(pending)+min(offset+20,len(pending))}/{len(ids)}", flush=True)
            time.sleep(.2)


def build_features(nodes):
    records = [json.loads((DATA/"semantic_sources"/f"{n['pageid']}.json").read_text()) for n in nodes]
    title = torch.from_numpy(np.stack([fast_title_features(n["title"], 128) for n in nodes]))
    # Bounded, fixed text hashing: these are not pretrained semantic embeddings.
    text = torch.from_numpy(np.stack([fast_title_features(r.get("intro", "")[:4000], 64) for r in records]))
    memberships = {}
    for i, r in enumerate(records):
        for category in r["categories"]:
            memberships.setdefault(category, []).append(i)
    # Broad groups are weak context; singleton groups do not share information.
    groups = {c: members for c, members in memberships.items() if 2 <= len(members) <= 100}
    context = torch.zeros_like(text)
    degree = torch.zeros(len(nodes))
    for members in groups.values():
        context[members] += text[members].mean(0)
        degree[members] += 1
    context /= degree.clamp_min(1).unsqueeze(1)
    context = nn.functional.normalize(context, dim=1)
    save_json(DATA/"hypergraph.json", {"type": "Undirected visible-category groups; fixed article-to-group-to-article text pooling",
        "hyperedges": [{"category": c, "members": [nodes[i]["pageid"] for i in members]} for c, members in sorted(groups.items())],
        "not_implemented": "No free-form relation extraction, pretrained language encoder, Coconut, or learned hypergraph message passing."})
    save_json(DATA/"semantic_manifest.json", {"articles": len(nodes), "intros_present": sum(bool(r.get("intro")) for r in records),
        "category_hyperedges": len(groups), "articles_in_hyperedges": int((degree > 0).sum()),
        "feature_scope": "Fixed hashes of title and intro; category hypergraph pools intro vectors. Unsupervised features use all graph articles (transductive); no distance labels used."})
    return title, torch.cat((text, context), dim=1)


def decisions(graph, goals, index, per_goal, seed):
    rows = []
    raw = {}
    for goal in goals:
        distances = nx.single_source_shortest_path_length(graph.reverse(copy=False), goal)
        eligible = sorted(a for a, distance in distances.items() if distance >= 2 and graph.out_degree(a) >= 2)
        random.Random(seed+goal).shuffle(eligible)
        for current in eligible[:per_goal]:
            neighbors = sorted(graph.successors(current))
            optimal = [distances.get(b) == distances[current]-1 for b in neighbors]
            assert any(optimal)
            rows.append({"current": index[current], "goal": index[goal], "candidates": [index[b] for b in neighbors],
                         "optimal": optimal, "distance": distances[current]})
            for b in neighbors:
                if b in distances:
                    raw[index[b], index[goal]] = distances[b]
    return rows, raw








@torch.inference_mode()
def evaluate_rank(model, rows, title, semantic, errors=False):
    model.eval()
    losses, correct, mistakes = [], 0, []
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        for j, (row, logits) in enumerate(zip(batch, grouped_batch(batch, title, semantic, model))):
            losses.append(ranking_loss(logits, row["optimal"]).item())
            okay = row["optimal"][int(logits.argmax())]
            correct += okay
            if not okay:
                mistakes.append(start+j)
    result = {"loss": float(np.mean(losses)), "optimal_next_link_accuracy": correct/len(rows), "decisions": len(rows)}
    return (result, mistakes) if errors else result


def train_rank(model, train, val, title, semantic, epochs, seed, output, weights=None, graph_id=None):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    best_state = copy.deepcopy(model.state_dict())
    initial = evaluate_rank(model, val, title, semantic)
    best, best_epoch = initial["loss"], 0
    history = [{"epoch": 0, "train_loss": None, **{"validation_"+k: v for k,v in initial.items()}}]
    for epoch in range(1, epochs+1):
        model.train()
        order = (torch.randperm(len(train), generator=generator) if weights is None else
                 torch.multinomial(weights, len(train), replacement=True, generator=generator)).tolist()
        total = 0.
        for start in range(0, len(order), 32):
            batch = [train[i] for i in order[start:start+32]]
            optimizer.zero_grad()
            logits = grouped_batch(batch, title, semantic, model)
            loss = torch.stack([ranking_loss(values, row["optimal"]) for row, values in zip(batch, logits)]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item()*len(batch)
        validation = evaluate_rank(model, val, title, semantic)
        history.append({"epoch": epoch, "train_loss": total/len(train), **{"validation_"+k: v for k,v in validation.items()}})
        if validation["loss"] < best:
            best, best_epoch, best_state = validation["loss"], epoch, copy.deepcopy(model.state_dict())
        print(f"{output.name} epoch {epoch}: train={total/len(train):.4f}, val={validation['loss']:.4f}, next-link={validation['optimal_next_link_accuracy']:.3f}", flush=True)
        save_json(output.with_suffix(".progress.json"), {"epoch": epoch, "selected_epoch": best_epoch})
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "semantic": model.semantic, "selected_epoch": best_epoch,
                "graph_id": graph_id or json.loads((DATA/"manifest.json").read_text())["graph_id"]}, output)
    save_json(output.with_suffix(".history.json"), history)
    return {"selected_epoch": best_epoch, "history": history}


def train_raw(pairs, val_pairs, title, epochs, seed):
    model = DistanceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    items = sorted(pairs.items())
    random.Random(seed).shuffle(items)
    # Same decision-derived labels; cap memory and CPU explicitly.
    items = items[:60000]
    indices = torch.tensor([key for key, label in items])
    labels = torch.tensor([float(label) for key, label in items])
    val_items = sorted(val_pairs.items())
    vi = torch.tensor([key for key, label in val_items])
    vy = torch.tensor([float(label) for key, label in val_items])
    generator = torch.Generator().manual_seed(seed)
    best, best_state, selected, history = float("inf"), None, 0, []
    for epoch in range(1, epochs+1):
        model.train(); total = 0.
        for batch in torch.randperm(len(items), generator=generator).split(256):
            ids = indices[batch]
            optimizer.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(pair_features(title[ids[:,0]], title[ids[:,1]])), labels[batch])
            loss.backward(); optimizer.step()
            total += loss.item()*len(batch)
        model.eval()
        with torch.inference_mode():
            total_error = 0.
            for start in range(0, len(vy), 1024):
                ids = vi[start:start+1024]
                total_error += (model(pair_features(title[ids[:,0]], title[ids[:,1]]))-vy[start:start+1024]).abs().sum().item()
            error = total_error/len(vy)
        history.append({"epoch": epoch, "train_loss": total/len(items), "validation_mae": error})
        if error < best:
            best, selected, best_state = error, epoch, copy.deepcopy(model.state_dict())
        print(f"raw epoch {epoch}: loss={total/len(items):.4f}, val MAE={error:.4f}", flush=True)
    return best_state, {"selected_epoch": selected, "history": history, "train_pairs": len(items)}


@torch.inference_mode()
def navigate(graph, queries, index, model, title, semantic, max_steps=20):
    model.eval()
    results = []
    for row in queries:
        current, goal = row["source"], row["target"]
        path, visited = [current], {current}
        started = time.perf_counter()
        status = "step_limit"
        for _ in range(max_steps):
            neighbors = sorted(b for b in graph.successors(current) if b not in visited)
            if goal in neighbors:
                path.append(goal); status = "found"; break
            if not neighbors:
                status = "dead_end"; break
            candidates = torch.tensor([index[b] for b in neighbors])
            goals = torch.full_like(candidates, index[goal])
            logits = model(candidates, goals, title, semantic)
            current = neighbors[int(logits.argmax())]
            path.append(current); visited.add(current)
        found = status == "found"
        results.append({**row, "status": status, "path": path if found else None, "visited_path": path,
                        "clicks": len(path)-1 if found else None, "seconds": time.perf_counter()-started})
    return results


def summarize(rows):
    success = [r for r in rows if r["path"] is not None]
    optimal = sum(r["clicks"] == r["optimal_clicks"] for r in success)
    return {"queries": len(rows), "successes": len(success), "optimal_routes": optimal,
            "success_rate": len(success)/len(rows), "optimal_rate": optimal/len(rows),
            "mean_extra_clicks_successes_only": float(np.mean([r["clicks"]-r["optimal_clicks"] for r in success])) if success else None,
            "median_ms": float(np.median([r["seconds"] for r in rows]))*1000}


def run(seeds, epochs, correction_epochs):
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    save_json(OUT/"status.json", {"state": "running", "seeds": seeds})
    graph, graph_id = load_graph(DATA)
    frozen = json.loads((DATA/"graph.json").read_text())
    manifest = json.loads((DATA/"manifest.json").read_text())
    nodes = sorted(frozen["nodes"], key=lambda n:n["pageid"])
    index = {n["pageid"]: i for i,n in enumerate(nodes)}
    title, semantic = build_features(nodes)
    splits = manifest["destination_splits"]
    assert not (set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"]) or set(splits["validation"]) & set(splits["test"]))
    train, raw_pairs = decisions(graph, splits["train"], index, 10, 613)
    val, raw_val = decisions(graph, splits["validation"], index, 10, 614)
    save_json(DATA/"train_decisions.json", train)
    save_json(DATA/"validation_decisions.json", val)
    summaries, models = {}, {}
    for seed in seeds:
        folder = OUT/f"seed_{seed}"; folder.mkdir(exist_ok=True)
        torch.manual_seed(seed); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        state, raw_info = train_raw(raw_pairs, raw_val, title, epochs, seed)
        torch.save({"state_dict": state, "dimensions":128, "hidden":64, "graph_id":graph_id}, folder/"raw.pt")
        save_json(folder/"raw.history.json", raw_info)
        raw = Ranker(state)
        title_rank = Ranker(state)
        train_rank(title_rank, train, val, title, semantic, epochs, seed, folder/"title_rank.pt")
        semantic_rank = Ranker(state, semantic=True)
        train_rank(semantic_rank, train, val, title, semantic, epochs, seed, folder/"semantic_rank.pt")
        initial, mistakes = evaluate_rank(semantic_rank, train, title, semantic, errors=True)
        weights = mistake_weights(len(train), mistakes)
        save_json(folder/"correction_examples.json", {"scope":"Training mistakes only, no validation/test mistakes", "train_decisions":len(train), "mistake_indices":mistakes, "weight":4, "before":initial})
        corrected, continued = copy.deepcopy(semantic_rank), copy.deepcopy(semantic_rank)
        # Both controls receive exactly the same number of optimizer updates.
        train_rank(continued, train, val, title, semantic, correction_epochs, seed+1000, folder/"continued.pt")
        train_rank(corrected, train, val, title, semantic, correction_epochs, seed+1000, folder/"corrected.pt", weights)
        models[seed] = {"raw":raw, "title_rank":title_rank, "semantic_rank":semantic_rank, "continued":continued, "corrected":corrected}
        summaries[str(seed)] = {"raw":raw_info, "correction_training_mistakes":len(mistakes)}
    # All model selection and error mining are complete before opening test labels.
    test, _ = decisions(graph, splits["test"], index, 10, 615)
    save_json(DATA/"test_decisions.json", test)
    queries = [{"source": nodes[r["current"]]["pageid"], "target": nodes[r["goal"]]["pageid"], "optimal_clicks":r["distance"]} for r in test]
    save_json(OUT/"evaluation_queries.json", queries)
    bfs_rows = []
    for q in queries:
        measured = search(graph, q["source"], q["target"], budget=len(graph))
        bfs_rows.append({**q, **measured})
    for seed, conditions in models.items():
        folder = OUT/f"seed_{seed}"
        report = {}
        for name, model in conditions.items():
            rows = navigate(graph, queries, index, model, title, semantic)
            report[name] = {"navigation":summarize(rows), "next_link":evaluate_rank(model, test, title, semantic)}
            save_json(folder/f"{name}.routes.json", rows)
        summaries[str(seed)]["test"] = report
        save_json(folder/"results.json", report)
    result = {"graph_id":graph_id, "nodes":len(graph), "edges":graph.number_of_edges(),
              "decision_counts":{"train":len(train), "validation":len(val), "test":len(test)},
              "seeds":seeds, "epochs_per_stage":epochs, "correction_epochs":correction_epochs,
              "bfs":summarize(bfs_rows), "runs":summaries, "elapsed_seconds":time.perf_counter()-started,
              "protocol":"Destination-disjoint splits; 20-click navigation with no revisits, direct target check shared by neural methods. BFS searches up to all nodes and is an optimality reference, not an equal-work navigation policy. Features precomputed for all methods; no live network calls during evaluation.",
              "scope":"Local pilot. Self-correction is offline supervised hard-example resampling, not autonomous fact correction, RL, test-time training or Coconut. Semantic context is fixed intro hashing and category-hypergraph pooling; additional parameters and text differ from raw title model."}
    save_json(OUT/"results.json", result)
    torch.save({"graph_id":graph_id, "title_vectors":title, "semantic_vectors":semantic, "page_ids":[n["pageid"] for n in nodes]}, OUT/"features.pt")
    save_json(OUT/"status.json", {"state":"complete", "elapsed_seconds":time.perf_counter()-started, "results":str(OUT/"results.json")})
    print(json.dumps({"counts":result["decision_counts"], "bfs":result["bfs"], "tests":{s:r["test"] for s,r in summaries.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fetch", "run"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[207,208,209])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--correction-epochs", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "fetch": fetch()
    else: run(args.seeds, args.epochs, args.correction_epochs)


if __name__ == "__main__":
    main()
