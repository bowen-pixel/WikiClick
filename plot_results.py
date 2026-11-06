"""Plot recorded training and paired search results without retraining models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"bfs": "#3066be", "text": "#d17a16", "neural": "#168577"}
LABELS = {"bfs": "BFS", "text": "Title similarity", "neural": "Neural guidance"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, folder, name):
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def training_curves(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for folder, label, color, style in [
        ("model_100", "100 articles", "#3066be", "--"),
        ("model_1100", "1,100 articles", "#168577", "-"),
    ]:
        metrics = read(runs / folder / "metrics.json")
        history = metrics["history"]
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["train_loss"] for h in history], style, color=color, label=label, linewidth=2)
        axes[1].plot(epochs, [h["validation_mae"] for h in history], style, color=color, label=label, linewidth=2)
        selected = metrics["selected_epoch"]
        error = next(h["validation_mae"] for h in history if h["epoch"] == selected)
        axes[1].scatter([selected], [error], color=color, s=45, zorder=4)
        axes[1].annotate(f"Selected epoch {selected}", (selected, error),
                         xytext=(.56, .58) if folder == "model_100" else (.25, .40),
                         textcoords="axes fraction", color=color,
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 1})
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Mean Smooth L1 loss (β = 1)")
    axes[1].set(title="Validation: choose the checkpoint here", xlabel="Epoch", ylabel="Mean absolute error (clicks)")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        ax.set_xlim(1, 25)
    fig.suptitle("Learning curves · recorded 25-epoch CPU runs", fontsize=14)
    save(fig, output, "training_curves")


def loss_shape(output):
    residual = np.linspace(-3, 3, 601)
    absolute = np.abs(residual)
    loss = np.where(absolute < 1, .5 * residual ** 2, absolute - .5)
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax.plot(residual, loss, color=COLORS["neural"], linewidth=2.5, label="Smooth L1, β = 1")
    ax.axvline(0, color="0.5", linewidth=.7)
    ax.axvline(-1, color="0.7", linestyle=":", linewidth=1)
    ax.axvline(1, color="0.7", linestyle=":", linewidth=1)
    ax.set(title="The loss function used during training", xlabel="Prediction − true remaining distance (clicks)", ylabel="Loss per example")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig, output, "loss_function")


def search_plots(report, output):
    summary = report["summary"]
    methods = ["bfs", "text", "neural"]
    labels = [LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    expansions = [summary[m]["mean_expanded_all_queries"] for m in methods]
    bars = axes[0, 0].bar(labels, expansions, color=colors, width=.6)
    axes[0, 0].bar_label(bars, fmt="%.2f", padding=4)
    axes[0, 0].set(title="Search work · lower is better", ylabel="Mean articles expanded (all queries)", ylim=(0, max(expansions)*1.2))
    optimal = np.array([summary[m]["optimal_route_rate_all_queries"]*100 for m in methods])
    success = np.array([summary[m]["success_rate"]*100 for m in methods])
    axes[0, 1].bar(labels, optimal, color="#168577", label="Optimal route")
    axes[0, 1].bar(labels, success-optimal, bottom=optimal, color="#d9a441", label="Longer route")
    axes[0, 1].bar(labels, 100-success, bottom=success, color="#c85b5b", label="Budget failure")
    for i, method in enumerate(methods):
        axes[0, 1].text(i, 104, f"{success[i]:.0f}% reached", ha="center", fontsize=9)
    axes[0, 1].set(title="Route quality · all queries", ylabel="Queries (%)", ylim=(0, 112))
    axes[0, 1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=9)
    runtime = [summary[m]["median_seconds_all_queries"]*1000 for m in methods]
    bars = axes[1, 0].bar(labels, runtime, color=colors, width=.6)
    axes[1, 0].bar_label(bars, fmt="%.3f", padding=4)
    axes[1, 0].set(title="Measured offline latency · lower is better", ylabel="Median search time (milliseconds)", ylim=(0, max(runtime)*1.2))
    budgets = [1, 2, 3, 5, 10, 20, 50, 100, report["budget_expansions"]]
    budgets = sorted(set(b for b in budgets if b <= report["budget_expansions"]))
    for method in methods:
        values = [r["methods"][method] for r in report["results"]]
        rates = [100*sum(v["path"] is not None and v["expanded"] <= b for v in values)/len(values) for b in budgets]
        axes[1, 1].plot(budgets, rates, marker="o", color=COLORS[method], label=LABELS[method])
    axes[1, 1].set(title="Success as the expansion budget grows", xlabel="Maximum expanded articles (log scale)", ylabel="Goals reached (%)", xscale="log", ylim=(0, 105))
    axes[1, 1].legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    n = len(report["results"])
    fig.suptitle(f"1,100-article graph · {n} matched held-out queries · budget {report['budget_expansions']}", fontsize=14)
    save(fig, output, "search_comparison")
    with (output / "search_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *next(iter(summary.values())).keys()])
        writer.writeheader()
        for method in methods:
            writer.writerow({"method": method, **summary[method]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    training_curves(args.runs, args.output)
    loss_shape(args.output)
    search_plots(read(args.runs / "comparison_1100.json"), args.output)
    print(f"Saved PNG/PDF charts and CSV summary to {args.output}")


if __name__ == "__main__":
    main()
