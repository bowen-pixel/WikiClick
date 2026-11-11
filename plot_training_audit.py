"""Audit saved experiments; score frozen and fine-tuned models on identical test pairs.

This reads existing checkpoints and labels. It does not train or choose checkpoints.
Run: .venv/bin/python plot_training_audit.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from large_training import pair_tensors
from ml_model import DistanceModel
from train_model import mean_absolute_error


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    output = Path("runs/training_audit")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    metrics = read("runs/model_large_trained/metrics.json")
    frozen = read("runs/large_frozen_comparison.json")
    trained = read("runs/large_trained_comparison.json")
    assert frozen["graph_id"] == trained["graph_id"] == metrics["graph_id"]
    assert [(r["source"], r["target"], r["optimal_clicks"]) for r in frozen["results"]] == [
        (r["source"], r["target"], r["optimal_clicks"]) for r in trained["results"]]
    assert frozen["budget_expansions"] == trained["budget_expansions"]
    assert frozen["seconds_per_query"] == trained["seconds_per_query"]
    vectors = np.load("data/cache/large_features/vectors.npy", mmap_mode="r")
    x_test, y_test = pair_tensors(Path("data/graphs/large_supervised/test.npz"), vectors)
    scores = {}
    for name, path in [("Frozen 1,100-article model", "runs/model_1100/best.pt"),
                       ("Fine-tuned model", "runs/model_large_trained/best.pt")]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = DistanceModel(checkpoint["dimensions"], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scores[name] = mean_absolute_error(model, x_test, y_test)
    assert abs(scores["Fine-tuned model"] - metrics["test_mae"]) < 1e-5
    train_labels = np.load("data/graphs/large_supervised/train.npz")["labels"]
    scores["Constant train-mean predictor"] = float((y_test-float(train_labels.mean())).abs().mean())
    reference = [r["optimal_clicks"] for r in trained["results"]]
    records = []
    for size, folder, report in [(100, "model_100", "comparison_100"),
                                  (1100, "model_1100", "comparison_1100"),
                                  (1001100, None, "fresh_million_comparison"),
                                  (2001100, "model_large_trained", "large_trained_comparison")]:
        m = read(f"runs/{folder}/metrics.json") if folder else None
        r = read(f"runs/{report}.json")
        records.append({"graph_articles": size,
                        "training_pairs": (m["examples"]["train"] if "examples" in m else m["pairs"]["train"]["selected_pairs"]) if m else 0,
                        "training": "fine-tuning" if size == 2001100 else "from scratch" if folder else "none: frozen transfer",
                        "test_mae": m["test_mae"] if m else None,
                        "constant_test_mae": m["constant_test_mae"] if m else None,
                        "neural": r["summary"]["neural"], "bfs": r["summary"]["bfs"]})
    audit = {"scope": "Cross-size tests differ; only the two-million frozen/fine-tuned comparison is matched.",
             "missing_training_sizes": [10, 1000, 1000000], "experiments": records,
             "matched_distance_test": {"graph_id": metrics["graph_id"], "pairs": len(y_test), "mae_clicks": scores},
             "matched_routing_test": {"queries": len(reference), "mean_optimal_clicks": float(np.mean(reference)),
                                      "budget": trained["budget_expansions"],
                                      "frozen": frozen["summary"], "fine_tuned": trained["summary"]}}
    (output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    bars = ax.barh(list(scores), list(scores.values()), color=["#58718b", "#168577", "#ba701a"])
    ax.bar_label(bars, fmt="%.3f clicks", padding=6)
    ax.invert_yaxis()
    ax.set(xlabel="Mean absolute error in remaining distance (lower is better)",
           xlim=(0, max(scores.values())*1.25),
           title=f"Same {len(y_test):,} held-out pairs on the 2,001,100-article graph")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.supxlabel("Fine-tuning helps relative to the frozen model, but the constant predictor has lower error.", fontsize=11)
    save(fig, output, "matched_distance_error")

    # Show the full failures as counts rather than drawing three overlapping zero curves.
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    labels = ["BFS", "Title similarity", "Frozen neural", "Fine-tuned neural"]
    outcomes = [trained["summary"]["bfs"], trained["summary"]["text"],
                frozen["summary"]["neural"], trained["summary"]["neural"]]
    failures = [s["queries"]-s["successes"] for s in outcomes]
    bars = ax.barh(labels, failures, color="#b55b55", height=.55)
    ax.bar_label(bars, labels=[f"{s['successes']} reached / {s['queries']} tested" for s in outcomes],
                 label_type="center", color="white")
    ax.set(xlim=(0, len(reference)*1.05), xlabel="Queries without a route at the search limit",
           title="Same 100 reachable queries: no routing gain demonstrated")
    ax.invert_yaxis()
    fig.supxlabel(f"200 expansions and 10 seconds per query; all failures exhausted expansions.\n"
                  f"True optimal paths average {np.mean(reference):.2f} clicks; failed searches have no route length.", fontsize=11)
    save(fig, output, "matched_search_outcomes")

    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", color="#168577", markersize=3)
    axes[0].set(title="Training loss keeps falling", ylabel="Mean Smooth L1 loss", ylim=(0, .32))
    axes[1].plot(epochs, [h["validation_mae"] for h in history], "o-", color="#58718b", markersize=3)
    axes[1].axvline(metrics["selected_epoch"], color="#ba701a", linestyle="--", label="Selected epoch 1")
    axes[1].set(title="Validation does not improve after epoch 1", ylabel="Validation MAE (clicks)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set(xlabel="Epoch", xlim=(.5, 25.5))
        ax.grid(alpha=.2)
    fig.suptitle("Actual fine-tuning on 405,522 labelled training pairs")
    fig.supxlabel(f"Before fine-tuning: validation MAE {metrics['initial_validation_mae']:.3f}. "
                  "Right axis zooms into epochs 1–25; test data did not select the checkpoint.", fontsize=10)
    save(fig, output, "fine_tuning_curves")
    lines = ["# WikiClick experiment audit", "", audit["scope"], "",
             "Articles are graph nodes, not training examples. Labels are sampled candidate/destination distances.", "",
             "| Graph articles | Training | Training pairs | Neural test MAE | Constant MAE | Neural routes found | BFS routes found |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for r in records:
        mae = f"{r['test_mae']:.3f}" if r["test_mae"] is not None else "Not measured"
        baseline = f"{r['constant_test_mae']:.3f}" if r["constant_test_mae"] is not None else "Not measured"
        lines.append(f"| {r['graph_articles']:,} | {r['training']} | {r['training_pairs']:,} | {mae} | {baseline} | {r['neural']['successes']}/100 | {r['bfs']['successes']}/100 |")
    lines += ["", "No saved training run exists for exactly 10, 1,000 or 1,000,000 articles.", "",
              "## Matched two-million comparison", "", f"Same {len(y_test):,} held-out distance pairs:"]
    lines += [f"- {name}: {value:.6f} clicks MAE." for name, value in scores.items()]
    lines += ["", "All four routing approaches reached 0/100 goals on the shared test queries within 200 expansions and 10 seconds. All stopped at the expansion budget. These pairs have known paths; this is a budget floor, not evidence that paths do not exist.", "",
              "The constant distance baseline is not a routing policy; beating it on MAE is a useful sanity check, but routing quality needs its own test.", "",
              "## Interpretation", "", "Small runs used supervised training from random initialization. The million run was frozen inference only. The two-million run fine-tuned the 1,100 model with real optimizer updates. No test-time training has been implemented.", "",
              "The model uses hashed article titles in a small MLP, not article text or graph structure. Falling training loss alone does not establish generalization or better routing. Lower validation error relative to the old model did not beat a constant on the test split or improve routing success.", "",
              "## Next controlled experiment", "", "Keep one graph and destination-disjoint validation/test sets fixed. Vary nested training-pair counts, train multiple seeds, and compare against the train-mean distance baseline and BFS. Report success, optimal-route rate, expansions and total latency across larger search budgets. Choose budgets on validation queries first; reserve new test queries after tuning. Test graph/text features or a neighbor-ranking objective before collecting more pages. Any test-time updates require a separately defined objective and an evaluation that includes update costs and excludes true test-distance labels.", "",
              "Rebuild these outputs with `.venv/bin/python plot_training_audit.py` from the repository directory."]
    Path("TRAINING_AUDIT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(audit["matched_distance_test"], indent=2))
    print(f"Saved report and charts in {output}")


if __name__ == "__main__":
    main()
