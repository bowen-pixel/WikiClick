"""Train a small CPU PyTorch distance model from saved WikiClick JSONL files."""

import argparse
import copy
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from codes.models.raw.ml_model import DistanceModel, FEATURE_VERSION, pair_features, title_features
from codes.api.sample_dataset import save_json


def load_pairs(path, graph_id, goals, limit, seed):
    """Deduplicate (candidate,destination): current article does not affect its label."""
    pairs = {}
    null_rows = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            destination = row["destination"]["pageid"]
            if row["graph_id"] != graph_id or row["split"] != path.stem or destination not in goals:
                raise ValueError(f"Graph/split mismatch in {path}")
            label = row["label_distance"]
            if label is None:
                null_rows += 1
                continue
            key = (row["candidate_next_article"]["pageid"], destination)
            if key in pairs and pairs[key] != label:
                raise ValueError(f"Conflicting distance labels for {key}")
            pairs[key] = label
    items = sorted(pairs.items())
    random.Random(seed).shuffle(items)
    selected = items[:limit]
    if not selected:
        raise ValueError(f"No reachable pairs in {path}")
    return selected, {"unique_reachable_pairs": len(items), "selected_pairs": len(selected), "null_rows_skipped": null_rows}


def make_tensors(pairs, vectors):
    x = torch.stack([pair_features(vectors[b], vectors[d]) for (b, d), label in pairs])
    y = torch.tensor([float(label) for key, label in pairs])
    return x, y


@torch.inference_mode()
def mean_absolute_error(model, x, y):
    total = 0.0
    for start in range(0, len(y), 1024):
        total += (model(x[start:start+1024]) - y[start:start+1024]).abs().sum().item()
    return total / len(y)


def train(data, output, epochs=25, max_pairs=30000, seed=207, dimensions=128, hidden=64):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    graph = json.loads((data / "graph.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    if graph["graph_id"] != manifest["graph_id"]:
        raise ValueError("Graph and manifest do not match")
    goals = {split: set(value["destination_ids"]) for split, value in manifest["training"]["splits"].items()}
    if goals["train"] & goals["validation"] or goals["train"] & goals["test"] or goals["validation"] & goals["test"]:
        raise ValueError("Destination splits overlap")
    vectors = {node["pageid"]: title_features(node["title"], dimensions) for node in graph["nodes"]}
    # Do not load test labels until training and model selection are finished.
    splits = {}
    counts = {}
    for split in ("train", "validation"):
        pairs, counts[split] = load_pairs(data / f"{split}.jsonl", graph["graph_id"], goals[split], max_pairs, seed)
        splits[split] = make_tensors(pairs, vectors)
    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = DistanceModel(dimensions, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()
    best_error = float("inf")
    best_state = None
    best_epoch = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        model.eval()
        error = mean_absolute_error(model, x_validation, y_validation)
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "validation_mae": error})
        print(f"Epoch {epoch:02d}: loss={total/len(y_train):.4f}, validation MAE={error:.4f}", flush=True)
        if error < best_error:
            best_error, best_epoch = error, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    checkpoint = {"state_dict": best_state, "dimensions": dimensions, "hidden": hidden,
                  "feature_version": FEATURE_VERSION, "graph_id": graph["graph_id"],
                  "training_graph_node_ids": sorted(vectors),
                  "destination_splits": {k: sorted(v) for k, v in goals.items()},
                  "seed": seed, "selected_epoch": best_epoch}
    torch.save(checkpoint, output / "best.pt")
    # One final test evaluation after selecting the checkpoint by validation MAE.
    test_pairs, counts["test"] = load_pairs(data / "test.jsonl", graph["graph_id"], goals["test"], max_pairs, seed)
    x_test, y_test = make_tensors(test_pairs, vectors)
    test_error = mean_absolute_error(model, x_test, y_test)
    constant = y_train.mean().item()
    metrics = {"graph_id": graph["graph_id"], "torch_version": str(torch.__version__),
               "device": "cpu", "feature_version": FEATURE_VERSION, "dimensions": dimensions,
               "hidden": hidden, "epochs": epochs, "seed": seed, "selected_epoch": best_epoch,
               "validation_mae": best_error, "test_mae": test_error,
               "constant_train_mean": constant,
               "constant_test_mae": (y_test-constant).abs().mean().item(),
               "training_seconds": time.perf_counter()-started, "pairs": counts,
               "history": history,
               "scope": "Reachable distance prediction only; title features; shared graph, held-out destinations. No optimality guarantee."}
    save_json(output / "metrics.json", metrics)
    print(f"Saved {output / 'best.pt'}; test MAE={test_error:.4f}, constant baseline MAE={metrics['constant_test_mae']:.4f}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/sample_100"))
    parser.add_argument("--output", type=Path, default=Path("models/raw/model_100"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-pairs", type=int, default=30000)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_pairs < 1:
        parser.error("epochs and max-pairs must be positive")
    train(args.data, args.output, epochs=args.epochs, max_pairs=args.max_pairs)


if __name__ == "__main__":
    main()
