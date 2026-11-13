"""Train distance heads, or compare SHA256 and semantic titles on saved graphs.

Example: python -m codes.models.raw.train_model --compare-features
Only generated models/results/figures are written; input datasets are read-only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from codes.models.raw.ml_model import (DistanceModel, FEATURE_VERSION, SEMANTIC_FEATURE_VERSION,
                                      ENCODER_NAME, pair_features, create_encoder, build_title_vectors)
from codes.api.sample_dataset import save_json


def load_pairs(path, graph_id, goals, limit=None, seed=207):
    """Deduplicate candidate/destination pairs; support the previous public API."""
    pairs, null_rows = {}, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            goal = row["destination"]["pageid"]
            if row["graph_id"] != graph_id or row["split"] != path.stem or goal not in goals:
                raise ValueError(f"Graph/split mismatch in {path}")
            label = row["label_distance"]
            if label is None:
                null_rows += 1
                continue
            key = (row["candidate_next_article"]["pageid"], goal)
            if key in pairs and pairs[key] != label:
                raise ValueError(f"Conflicting distance labels for {key}")
            pairs[key] = label
    return select_pairs(sorted(pairs.items()), limit, seed, null_rows)


def select_pairs(items, limit, seed, null_rows=0):
    total = len(items)
    random.Random(seed).shuffle(items)
    selected = items if limit is None else items[:limit]
    if not selected:
        raise ValueError("No reachable pairs")
    digest = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
    return selected, {"unique_reachable_pairs": total, "selected_pairs": len(selected),
                      "null_rows_skipped": null_rows, "selected_pair_hash": digest}


def read_graph(data):
    graph = json.loads((data/"graph.json").read_text())
    manifest = json.loads((data/"manifest.json").read_text())
    if graph["graph_id"] != manifest["graph_id"]:
        raise ValueError("Graph and manifest mismatch")
    source = manifest["training"]["splits"] if "training" in manifest else manifest["destination_splits"]
    goals = {name: set(info["destination_ids"] if isinstance(info, dict) else info) for name, info in source.items()}
    sets = [goals[name] for name in ("train", "validation", "test")]
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError("Destination splits overlap")
    network = nx.DiGraph()
    network.add_nodes_from((n["pageid"], {"title": n["title"]}) for n in graph["nodes"])
    if len(network) != len(graph["nodes"]):
        raise ValueError("Duplicate graph node IDs")
    for edge in graph["edges"]:
        if edge.get("weight", 1) != 1 or edge["source"] not in network or edge["target"] not in network:
            raise ValueError("Invalid graph edge")
        network.add_edge(edge["source"], edge["target"], weight=1)
    return graph, goals, network


def split_pairs(data, split, graph, goals, network, max_pairs=30000, seed=207):
    path = data/f"{split}.jsonl"
    if path.exists():
        return load_pairs(path, graph["graph_id"], goals[split], max_pairs, seed)
    # The saved EXACT 1,000-node graph stores ranking decisions rather than raw
    # JSONL distance rows. Derive regression labels in memory on that same graph;
    # do not quietly substitute the older 1,100-node dataset or write new data.
    items, unreachable = [], 0
    reverse = network.reverse(copy=False)
    for goal in sorted(goals[split]):
        distances = nx.single_source_shortest_path_length(reverse, goal)
        unreachable += len(network)-len(distances)
        items.extend(((candidate, goal), distance) for candidate, distance in sorted(distances.items()))
    return select_pairs(items, max_pairs, seed, unreachable)


def make_tensors(pairs, vectors):
    x = torch.stack([pair_features(vectors[b], vectors[d]) for (b, d), _ in pairs])
    y = torch.tensor([float(label) for _, label in pairs], dtype=torch.float32)
    return x, y


@torch.inference_mode()
def mean_absolute_error(model, x, y):
    if not len(y):
        raise ValueError("MAE requires at least one example")
    total = 0.
    for i in range(0, len(y), 1024):
        j = i+1024
        total += (model(x[i:j])-y[i:j]).abs().sum().item()
    return total/len(y)


def prepare_data(data, dimensions, feature_method="semantic", encoder=None):
    graph, goals, _ = read_graph(data)
    vectors = build_title_vectors(graph, encoder, dimensions, feature_method)
    return graph["graph_id"], goals, vectors


def load_split(data, split, graph_id, goals, vectors, max_pairs=30000, seed=207):
    graph, actual_goals, network = read_graph(data)
    if graph["graph_id"] != graph_id or actual_goals != goals:
        raise ValueError("Graph/split mismatch")
    pairs, counts = split_pairs(data, split, graph, goals, network, max_pairs, seed)
    return (*make_tensors(pairs, vectors), counts)


def create_model(dimensions, hidden):
    model = DistanceModel(dimensions, hidden)
    return model, torch.optim.Adam(model.parameters(), lr=.001), nn.SmoothL1Loss()


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total, count = 0., 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()*len(y)
        count += len(y)
    return total/count


@torch.inference_mode()
def evaluate_distance(model, x, y, loss_fn):
    model.eval()

    total_loss = 0.0
    total_error = 0.0
    batch_size = 1024

    for start in range(0, len(y), batch_size):
        end = start + batch_size

        x_batch = x[start:end]
        y_batch = y[start:end]

        predictions = model(x_batch)

        loss = loss_fn(predictions, y_batch)
        errors = (predictions - y_batch).abs()

        total_loss += loss.item() * len(y_batch)
        total_error += errors.sum().item()

    return {
        "smooth_l1": total_loss / len(y),
        "mae": total_error / len(y)
    }




def fit_model(
    model,
    loader,
    optimizer,
    loss_fn,
    x_validation,
    y_validation,
    epochs,
    early_stop_patience=15,
    min_delta=0.0001
):
    best_error = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # Access the full training tensors for evaluation.
    x_train, y_train = loader.dataset.tensors

    for epoch in range(1, epochs + 1):

        # 1. Train all mini-batches.
        train_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn
        )

        # 2. Evaluate both partitions.
        train_metrics = evaluate_distance(
            model, x_train, y_train, loss_fn
        )

        validation_metrics = evaluate_distance(
            model, x_validation, y_validation, loss_fn
        )

        error = validation_metrics["mae"]

        current_lr = optimizer.param_groups[0]["lr"]

        # 4. Record comparable metrics.
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["smooth_l1"],
            "train_mae": train_metrics["mae"],
            "validation_loss": validation_metrics["smooth_l1"],
            "validation_mae": error,
            "learning_rate": current_lr,
            "optimizer_updates": epoch * len(loader)
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Val MAE: {error:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True
        )

        # 5. Select the best checkpoint.
        if error < best_error - min_delta:
            best_error = error
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 6. Stop when validation no longer improves.
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best-performing weights.
    model.load_state_dict(best_state)

    return best_state, best_epoch, best_error, history



def fit_prepared(train_pairs, validation_pairs, vectors, dimensions, hidden, epochs, seed, use_scheduler=False):
    # The frozen encoder may consume random numbers. Reset AFTER feature generation
    # so BOTH arms really start with identical neural weights and batch order.
    torch.manual_seed(seed)
    x_train, y_train = make_tensors(train_pairs, vectors)
    x_val, y_val = make_tensors(validation_pairs, vectors)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model, optimizer, loss_fn = create_model(dimensions, hidden)
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    started = time.perf_counter()
    state, epoch, error, history = fit_model(
    model,
    loader,
    optimizer,
    loss_fn,
    x_val,
    y_val,
    epochs)
    return model, {"state": state, "selected_epoch": epoch, "validation_mae": error, "history": history,
                   "training_seconds": time.perf_counter()-started, "constant_train_mean": y_train.mean().item(),
                   "initial_weight_hash": initial_hash}


def evaluate_and_save(model, fit, test_pairs, vectors, graph, goals, counts, output,
                      dimensions, hidden, epochs, seed, feature_method, feature_seconds=0., encoder_revision=None):
    output.mkdir(parents=True, exist_ok=True)
    x_test, y_test = make_tensors(test_pairs, vectors)
    model.eval()
    version = FEATURE_VERSION if feature_method == "sha256" else SEMANTIC_FEATURE_VERSION
    metrics = {"graph_id": graph["graph_id"], "graph_nodes": len(graph["nodes"]), "torch_version": str(torch.__version__),
               "device": "cpu", "feature_method": feature_method, "feature_version": version,
               "encoder_name": ENCODER_NAME if feature_method == "semantic" else None,
               "encoder_revision": encoder_revision if feature_method == "semantic" else None,
               "dimensions": dimensions, "hidden": hidden, "epochs": epochs, "seed": seed,
               **{k: v for k, v in fit.items() if k != "state"}, "feature_seconds": feature_seconds,
               "test_mae": mean_absolute_error(model, x_test, y_test),
               "constant_test_mae": (y_test-fit["constant_train_mean"]).abs().mean().item(), "pairs": counts,
               "scope": "Frozen title features, identical supervised distance head; destination-held-out evaluation in saved graph."}
    torch.save({"state_dict": fit["state"], "dimensions": dimensions, "hidden": hidden,
                "feature_version": version, "feature_method": feature_method,
                "encoder_name": metrics["encoder_name"], "encoder_revision": metrics["encoder_revision"],
                "graph_id": graph["graph_id"], "training_graph_node_ids": sorted(vectors),
                "destination_splits": {k: sorted(v) for k, v in goals.items()}, "seed": seed,
                "selected_epoch": fit["selected_epoch"]}, output/"best.pt")
    save_json(output/"metrics.json", metrics)
    return metrics


def train(
    data,
    output,
    epochs=25,
    max_pairs=30000,
    seed=207,
    dimensions=128,
    hidden=64,
    feature_method="sha256",
    encoder=None,
    use_scheduler=False):

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    graph, goals, network = read_graph(data)
    started = time.perf_counter()
    vectors = build_title_vectors(graph, encoder, dimensions, feature_method)
    feature_seconds = time.perf_counter()-started
    counts, pairs = {}, {}
    for split in ["train", "validation"]:
        pairs[split], counts[split] = split_pairs(data, split, graph, goals, network, max_pairs, seed)
    model, fit = fit_prepared(
    pairs["train"],
    pairs["validation"],
    vectors,
    dimensions,
    hidden,
    epochs,
    seed,
    use_scheduler=use_scheduler)
    # Test labels are loaded only after selecting the checkpoint.
    pairs["test"], counts["test"] = split_pairs(data, "test", graph, goals, network, max_pairs, seed)
    return evaluate_and_save(model, fit, pairs["test"], vectors, graph, goals, counts, output,
                             dimensions, hidden, epochs, seed, feature_method, feature_seconds)


@torch.inference_mode()
def route_results(model, vectors, network, queries, budget):
    from codes.search.compare_search import search
    model.eval()
    def score(nodes, goal):
        candidates = torch.stack([vectors[b] for b in nodes])
        destinations = vectors[goal].expand_as(candidates)
        return model(pair_features(candidates, destinations)).tolist()
    rows = []
    for query in queries:
        row = {**query, **search(network, query["source"], query["target"], budget, score)}
        if row["path"] is not None:
            assert all(network.has_edge(a, b) for a, b in zip(row["path"], row["path"][1:]))
            assert row["clicks"] >= row["optimal_clicks"]
        rows.append(row)
    successful = [r for r in rows if r["path"] is not None]
    summary = {"queries": len(rows), "success_rate": len(successful)/len(rows),
               "optimal_rate": sum(r["clicks"] == r["optimal_clicks"] for r in successful)/len(rows),
               "mean_expanded": sum(r["expanded"] for r in rows)/len(rows),
               "median_ms": float(torch.tensor([r["seconds"]*1000 for r in rows]).median()),
               "mean_extra_clicks_successes_only": sum(r["clicks"]-r["optimal_clicks"] for r in successful)/len(successful) if successful else None}
    return {"summary": summary, "results": rows}


def comparison_plots(report, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"sha256": "#54738d", "semantic": "#168577"}
    labels = {"sha256": "SHA256 title features", "semantic": "EmbeddingGemma 2 titles"}
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for extension in ["png", "pdf"]:
            fig.savefig(figures/f"{name}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    for dataset in report["datasets"]:
        runs = dataset["runs"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for method in ["sha256", "semantic"]:
            histories = [r[method]["history"] for r in runs.values()]
            epochs = [h["epoch"] for h in histories[0]]
            for ax, field in zip(axes, ["train_loss", "validation_mae"]):
                values = np.array([[h[field] for h in history] for history in histories])
                ax.plot(epochs, values.mean(0), label=labels[method], color=colors[method], linewidth=2)
                ax.fill_between(epochs, values.min(0), values.max(0), color=colors[method], alpha=.15)
        axes[0].set(title="Training loss", ylabel="Mean Smooth L1 loss")
        axes[1].set(title="Held-out validation distance error", ylabel="MAE (clicks; lower is better)")
        for ax in axes:
            ax.set_xlabel("Epoch"); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=10)
        fig.suptitle(f"{dataset['nodes']:,} articles · matched {report['dimensions']}d features and neural heads")
        fig.supxlabel("Lines: seed means; shaded areas: min–max across seeds. No smoothing. Test labels do not select epochs.", fontsize=10)
        save(fig, f"learning_curves_{dataset['nodes']}")
    fig, axes = plt.subplots(1, len(report["datasets"]), figsize=(6*len(report["datasets"]), 5), squeeze=False, layout="constrained")
    aggregate = []
    for ax, dataset in zip(axes[0], report["datasets"]):
        values = [[r[m]["test_mae"] for r in dataset["runs"].values()] for m in ["sha256", "semantic"]]
        constants = [r["sha256"]["constant_test_mae"] for r in dataset["runs"].values()]
        means = [float(np.mean(v)) for v in values]
        bars = ax.bar([0, 1], means, color=list(colors.values()), width=.55)
        ax.bar_label(bars, fmt="%.3f", padding=5)
        for i, v in enumerate(values): ax.scatter(np.full(len(v), i), v, color="black", s=20, zorder=3)
        ax.axhline(np.mean(constants), color="#ba701a", linestyle="--", label="Constant train-mean baseline")
        ax.set(xticks=[0, 1], xticklabels=["SHA256", "Semantic"], title=f"{dataset['nodes']:,} articles",
               ylabel="Test MAE (clicks; lower is better)", ylim=(0, max(max(v) for v in values+ [constants])*1.25))
        ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        gain = 100*(means[0]-means[1])/means[0]
        aggregate.append({"dataset": dataset["dataset"], "nodes": dataset["nodes"], "sha256_test_mae": means[0],
                          "semantic_test_mae": means[1], "relative_mae_reduction_percent": gain,
                          "constant_test_mae": float(np.mean(constants))})
    fig.suptitle("Final test: same pair identities, splits, seeds and head size")
    save(fig, "test_mae_comparison")
    fig, axes = plt.subplots(1, len(report["datasets"]), figsize=(6*len(report["datasets"]), 5), squeeze=False, layout="constrained")
    for ax, dataset in zip(axes[0], report["datasets"]):
        for i, (method, color) in enumerate(colors.items()):
            summaries = [r[method]["routes"]["summary"] for r in dataset["runs"].values()]
            success = 100*np.mean([s["success_rate"] for s in summaries]); optimal = 100*np.mean([s["optimal_rate"] for s in summaries])
            bars = ax.bar([i-.17, i+.17], [success, optimal], color=color, alpha=.85, width=.32)
            ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
            bars[1].set_hatch("//")
        ax.set(xticks=[0, 1], xticklabels=["SHA256", "Semantic"], ylim=(0, 112), title=f"{dataset['nodes']:,} articles", ylabel="Percent of matched route queries")
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("Route search: solid = goal reached; hatched = minimum-click route")
    fig.supxlabel(f"Identical best-first search · {report['route_budget']} expansions · fixed per-graph query set · precomputed features", fontsize=10)
    save(fig, "route_comparison")
    return aggregate


def compare_features(datasets, output, figures, epochs=25, seeds=(207, 208, 209), dimensions=256,
                     hidden=64, max_pairs=30000, queries=100, budget=200, encoder_revision=None):
    if (output/"comparison.json").exists():
        raise FileExistsError("This comparison already exists; choose a different --output and --figures directory")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    contexts = [(Path(path), *read_graph(Path(path))) for path in datasets]
    union = {}
    for _, graph, _, _ in contexts:
        for node in graph["nodes"]:
            if node["pageid"] in union and union[node["pageid"]]["title"] != node["title"]:
                raise ValueError("Same page ID has differing titles across graphs")
            union[node["pageid"]] = node
    union_graph = {"nodes": [union[i] for i in sorted(union)]}
    feature_maps, feature_times = {}, {}
    for method in ["sha256", "semantic"]:
        started = time.perf_counter()
        encoder = create_encoder(revision=encoder_revision, cache_folder="data/cache/semantic_encoders") if method == "semantic" else None
        feature_maps[method] = build_title_vectors(union_graph, encoder, dimensions, method)
        feature_times[method] = time.perf_counter()-started
        if encoder is not None:
            encoder_revision = getattr(getattr(encoder[0].auto_model, "config", None), "_commit_hash", None) or encoder_revision
            del encoder
    report = {"dimensions": dimensions, "hidden": hidden, "epochs": epochs, "seeds": list(seeds),
              "max_pairs_per_split": max_pairs, "route_budget": budget, "encoder": ENCODER_NAME,
              "encoder_revision": encoder_revision, "feature_seconds_for_union": feature_times, "datasets": [],
              "protocol": "Frozen title encoders; matched dimensions, head, initialization, training pairs/order and split per seed. Validation selects epochs; final test reports distance MAE and fixed-query search. Encoding costs separate from indexed search. Different graph sizes are separate tasks."}
    from codes.search.compare_search import choose_queries
    for data, graph, goals, network in contexts:
        dataset = {"dataset": str(data), "graph_id": graph["graph_id"], "nodes": len(network), "edges": network.number_of_edges(), "runs": {}}
        vectors = {m: {n: feature_maps[m][n] for n in network} for m in feature_maps}
        for seed in seeds:
            print(f"\nDATASET {len(network)} articles, seed {seed}", flush=True)
            pairs, counts, fits = {}, {}, {}
            for split in ["train", "validation"]:
                pairs[split], counts[split] = split_pairs(data, split, graph, goals, network, max_pairs, seed)
            for method in ["sha256", "semantic"]:
                print(f"Training {method}", flush=True)
                fits[method] = fit_prepared(pairs["train"], pairs["validation"], vectors[method], dimensions, hidden, epochs, seed)
            assert fits["sha256"][1]["initial_weight_hash"] == fits["semantic"][1]["initial_weight_hash"]
            pairs["test"], counts["test"] = split_pairs(data, "test", graph, goals, network, max_pairs, seed)
            # Fixed across seeds as well as methods; references guide evaluation only.
            query_rows = choose_queries(network, goals["test"], queries, 902)
            record = {}
            for method in ["sha256", "semantic"]:
                model, fit = fits[method]
                folder = output/f"articles_{len(network)}"/f"seed_{seed}"/method
                metrics = evaluate_and_save(model, fit, pairs["test"], vectors[method], graph, goals, counts, folder,
                                            dimensions, hidden, epochs, seed, method, encoder_revision=encoder_revision)
                metrics["routes"] = route_results(model, vectors[method], network, query_rows, budget)
                save_json(folder/"metrics.json", metrics)
                record[method] = metrics
            dataset["runs"][str(seed)] = record
            save_json(output/"progress.json", {"dataset_nodes": len(network), "completed_seed": seed})
        report["datasets"].append(dataset)
    save_json(output/"comparison.json", report)
    aggregate = comparison_plots(report, figures)
    save_json(output/"summary.json", aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Figures: {figures.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/sample_100"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--features", choices=["sha256", "semantic"], default="sha256")
    parser.add_argument("--compare-features", action="store_true")
    parser.add_argument("--datasets", type=Path, nargs="+", default=[Path("data/graphs/sample_100"), Path("data/graphs/ranking_1000")])
    parser.add_argument("--figures", type=Path, default=Path("figures/raw/title_feature_comparison"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[207, 208, 209])
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--encoder-revision")
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.max_pairs, args.queries, args.budget) < 1 or not args.seeds:
        parser.error("Epochs, hidden size, pairs, queries and budget must be positive")
    if args.dimensions not in (128, 256, 512, 768):
        parser.error("Choose 128, 256, 512 or 768 feature dimensions")
    if args.compare_features:
        compare_features(args.datasets, args.output or Path("models/raw/title_feature_comparison"), args.figures,
                         args.epochs, args.seeds, args.dimensions, args.hidden, args.max_pairs, args.queries,
                         args.budget, args.encoder_revision)
    else:
        output = args.output or Path("models/raw")/f"{args.features}_{args.data.name}"
        if (output/"best.pt").exists():
            parser.error("Checkpoint already exists; choose a new --output to preserve it")
        encoder = create_encoder(revision=args.encoder_revision, cache_folder="data/cache/semantic_encoders") if args.features == "semantic" else None
        train(
    data=args.data,
    output=output,
    epochs=args.epochs,
    max_pairs=args.max_pairs,
    seed=args.seeds[0],
    dimensions=args.dimensions,
    hidden=args.hidden,
    feature_method=args.features,
    encoder=encoder)


if __name__ == "__main__":
    main()
