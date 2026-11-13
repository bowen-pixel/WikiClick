"""Controlled WikiClick regularization study: 4 settings x fixed model seeds.

Place at codes/models/raw/regularization_experiment.py and run from project root:
  python3 -m codes.models.raw.regularization_experiment --data data/graphs/ranking_1000

Feature vectors and sampled candidate/destination pairs are frozen across all arms.
Test labels are NEVER read by this script. Previous checkpoints are never overwritten.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from codes.models.raw.ml_model import (
    DistanceModel, ENCODER_NAME, SEMANTIC_FEATURE_VERSION,
    build_title_vectors, create_encoder,
)
from codes.models.raw.train_model import (
    fit_model, make_tensors, read_graph, split_pairs,
)



class RegularizedDistanceModel(DistanceModel):
    def __init__(
        self,
        dimensions=256,
        hidden=64,
        dropout1=0.0,
        dropout2=0.0
    ):
        super().__init__(dimensions, hidden)

        self.dropout1 = dropout1
        self.dropout2 = dropout2

        self.network = nn.Sequential(
            nn.Linear(dimensions * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(32, 1),
            nn.Softplus()
        )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_one_setting(
    x_train, y_train, x_val, y_val,
    dimensions, hidden, epochs, model_seed,
    dropout1, dropout2, weight_decay,
    early_stop_patience
):
    # Reinitialize the model identically for every arm with the same seed.
    torch.manual_seed(model_seed)
    model = RegularizedDistanceModel(dimensions, hidden, dropout1=dropout1,
    dropout2=dropout2)
    initial_hash = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.parameters()
    )).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(model_seed),
    )
    start = time.perf_counter()
    best_state, best_epoch, best_mae, history = fit_model(
        model, loader, optimizer, loss_fn, x_val, y_val,
        epochs=epochs,
        early_stop_patience=early_stop_patience, min_delta=0.0001,
    )
    return model, best_state, {
        "model_seed": model_seed,
        "dropout1": dropout1,
        "dropout2": dropout2,
        "weight_decay": weight_decay,
        "initial_weight_hash": initial_hash,
        "selected_epoch": best_epoch,
        "validation_mae": best_mae,
        "completed_epochs": len(history),
        "optimizer_updates": len(history) * len(loader),
        "training_seconds": time.perf_counter() - start,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/ranking_1000"))
    parser.add_argument("--output", type=Path, default=Path("models/experiments/regularization_1000"))
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--data-seed", type=int, default=611)
    parser.add_argument("--seeds", nargs="+", type=int, default=[207, 208, 209])
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--encoder-revision")
    args = parser.parse_args()
    if args.max_pairs < 1 or args.hidden < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("Positive values required for pairs, width, epochs, patience")
    if args.dimensions not in (128, 256, 512, 768):
        parser.error("Semantic dimensions must be 128, 256, 512, or 768")
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        parser.error("Model seeds must be distinct")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose another --output (never overwrite runs)")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    graph, goals, network = read_graph(args.data)
    pairs = {}
    counts = {}
    for split in ("train", "validation"):
        pairs[split], counts[split] = split_pairs(
            args.data, split, graph, goals, network, args.max_pairs, args.data_seed
        )
    if set(pairs["train"]) & set(pairs["validation"]):
        raise ValueError("Train/validation pair identities overlap")
    print("Generating frozen EmbeddingGemma title vectors once...", flush=True)
    encoder = create_encoder(revision=args.encoder_revision, cache_folder="data/cache/semantic_encoders")
    vectors = build_title_vectors(graph, encoder, args.dimensions, "semantic")
    del encoder
    x_train, y_train = make_tensors(pairs["train"], vectors)
    x_val, y_val = make_tensors(pairs["validation"], vectors)
    
# (dropout1, dropout2, weight_decay)
    settings = {
    "R0_baseline":       (0.00, 0.00, 0.0),
    "R1_dropout":        (0.10, 0.00, 0.0),
    "R2_weight_decay":   (0.00, 0.00, 0.0001),
    "R3_combined":       (0.10, 0.00, 0.0001),

    "R4_mild_increase":  (0.05, 0.10, 0.0),
    "R5_strong_increase":(0.10, 0.20, 0.0),
    "R6_decrease":       (0.10, 0.05, 0.0),
    "R7_strong_dropout": (0.20, 0.20, 0.0),
    }

    config = {
        "graph_id": graph["graph_id"], "graph_nodes": len(graph["nodes"]),
        "feature_method": "semantic", "feature_version": SEMANTIC_FEATURE_VERSION,
        "encoder_name": ENCODER_NAME, "encoder_revision_requested": args.encoder_revision,
        "dimensions": args.dimensions, "hidden": args.hidden,
        "data_seed": args.data_seed, "model_seeds": args.seeds,
        "max_pairs_per_split": args.max_pairs, "epochs_max": args.epochs,
        "early_stop_patience": args.patience, "learning_rate": 0.001,
        "use_scheduler": False, "batch_size": 256,
        "counts": counts, "settings": {
    k: {
        "dropout1": p1,
        "dropout2": p2,
        "weight_decay": wd
    }
    for k, (p1, p2, wd) in settings.items()
},
        "selection_metric": "held-out validation MAE (destination-disjoint)",
        "test_policy": "No test labels loaded or evaluated; lock a configuration before final test",
    }
    args.output.mkdir(parents=True, exist_ok=False) if not args.output.exists() else None
    write_json(args.output / "config.json", config)

    all_results = {}
    for seed in args.seeds:
        all_results[str(seed)] = {}
        initial_hashes = []
        for name, (dropout1, dropout2, wd) in settings.items():
            print(f"\n=== Model seed {seed} | {name} | dropout1={dropout1}| dropout2 = {dropout2} wd={wd} ===", flush=True)
            model, state, metrics = train_one_setting(
    x_train, y_train, x_val, y_val,
    args.dimensions, args.hidden,
    args.epochs, seed,
    dropout1, dropout2, wd,
    args.patience
)
            initial_hashes.append(metrics["initial_weight_hash"])
            folder = args.output / f"seed_{seed}" / name
            folder.mkdir(parents=True, exist_ok=False)
            torch.save({
        "state_dict": state,
        "dimensions": args.dimensions,
        "hidden": args.hidden,
        "dropout1": dropout1,
        "dropout2": dropout2,
        "weight_decay": wd,
        "feature_method": "semantic",
        "feature_version": SEMANTIC_FEATURE_VERSION,
        "encoder_name": ENCODER_NAME,
        "graph_id": graph["graph_id"],
        "model_seed": seed,
        "data_seed": args.data_seed,
        "selected_epoch": metrics["selected_epoch"],
        }, folder / "best.pt")
            write_json(folder / "metrics.json", metrics)
            all_results[str(seed)][name] = {
                "validation_mae": metrics["validation_mae"],
                "selected_epoch": metrics["selected_epoch"],
                "completed_epochs": metrics["completed_epochs"],
                "initial_weight_hash": metrics["initial_weight_hash"],
            }
            write_json(args.output / "progress.json", all_results)
        if len(set(initial_hashes)) != 1:
            raise AssertionError("Initial weights differ across arms for the same seed")

    averages = {}
    for name in settings:
        errors = [all_results[str(seed)][name]["validation_mae"] for seed in args.seeds]
        averages[name] = {"mean_validation_mae": sum(errors) / len(errors),
                          "per_seed_validation_mae": errors}
    winner = min(averages, key=lambda name: averages[name]["mean_validation_mae"])
    report = {"config": config, "results": all_results,
              "summary": averages, "provisional_validation_winner": winner}
    write_json(args.output / "comparison.json", report)
    print("\nValidation-only comparison (lower MAE is better):", flush=True)
    for name, entry in averages.items():
        print(f"  {name}: {entry['mean_validation_mae']:.6f} | {entry['per_seed_validation_mae']}")
    print(f"Provisional winner: {winner}. Do NOT choose based on test metrics.")


if __name__ == "__main__":
    main()
