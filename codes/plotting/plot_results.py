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
        fig.savefig(folder / f"{name}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def training_curves(runs, output, metric_paths=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    if metric_paths:
        records = [(path, (f"{read(path)['graph_nodes']:,} articles" if read(path).get("graph_nodes") else path.parent.name), color, style)
                   for path, color, style in zip(metric_paths, ["#168577", "#3066be", "#d17a16"], ["-", "--", ":"])]
    else:
        records = [(runs / "model_100/metrics.json", "100 articles", "#3066be", "--"),
                   (runs / "model_1100/metrics.json", "1,100 articles", "#168577", "-")]
    maximum_epoch = 1
    minimum_epoch = 1
    for index, (path, label, color, style) in enumerate(records):
        metrics = read(path)
        history = metrics["history"]
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["train_loss"] for h in history], style, color=color, label=label, linewidth=2)
        axes[1].plot(epochs, [h["validation_mae"] for h in history], style, color=color, label=label, linewidth=2)
        selected = metrics["selected_epoch"]
        error = (metrics["initial_validation_mae"] if selected == 0 else next(h["validation_mae"] for h in history if h["epoch"] == selected))
        maximum_epoch = max(maximum_epoch, max(epochs))
        minimum_epoch = min(minimum_epoch, selected)
        axes[1].scatter([selected], [error], color=color, s=45, zorder=4)
        axes[1].annotate(f"Selected epoch {selected}", (selected, error),
                         xytext=(.56, .58) if index == 0 else (.25, .40),
                         textcoords="axes fraction", color=color,
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 1})
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Mean Smooth L1 loss (β = 1)")
    axes[1].set(title="Validation: choose the checkpoint here", xlabel="Epoch", ylabel="Mean absolute error (clicks)")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        ax.set_xlim(minimum_epoch, max(2, maximum_epoch))
    fig.suptitle("Learning curves · recorded CPU training", fontsize=14)
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
    x = np.arange(len(methods))
    n = len(report["results"])
    node_count = report.get("graph_nodes", 1100)
    caption = f"{node_count:,} articles · {n} paired queries · {report['budget_expansions']} expansions"
    if report.get("seconds_per_query"):
        caption += f" · {report['seconds_per_query']:g}s/query"
    def panel(ax, index):
        if index == 0:
            values = [summary[m]["mean_expanded_all_queries"] for m in methods]
            bars = ax.bar(x, values, color=colors, width=.6)
            ax.bar_label(bars, fmt="%.2f", padding=5)
            ax.set(title="Search work", ylabel="Mean articles expanded (all queries)",
                   ylim=(0, max(1, max(values))*1.2))
        elif index == 1:
            success = np.array([summary[m]["success_rate"]*100 for m in methods])
            optimal = np.array([summary[m]["optimal_route_rate_all_queries"]*100 for m in methods])
            if not success.any():
                ax.axis("off")
                ax.set_title("No routes found within the search limits")
                table = ax.table(cellText=[[label, f"0 / {n}", "Not available"] for label in labels],
                                 colLabels=["Method", "Goals reached", "Route length"],
                                 cellLoc="center", bbox=[0, .35, 1, .4])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                ax.text(.5, .17, "All tested pairs have a known path.\nNo successful route means no route-length estimate.",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11)
                return
            bars = ax.bar(x-.17, success, width=.34, color=colors, label="Reached goal")
            ax.bar_label(bars, fmt="%.0f%%", padding=5)
            bars = ax.bar(x+.17, optimal, width=.34, color=colors, alpha=.4,
                          hatch="//", edgecolor="0.35", label="Optimal route")
            ax.bar_label(bars, fmt="%.0f%%", padding=5)
            ax.set(title="Success and optimal-route rates", ylabel="All evaluation queries (%)",
                   ylim=(0, min(112, max(5, max(success)*1.25))))
            ax.legend(frameon=False, loc="upper left", fontsize=10)
        elif index == 2:
            values = [summary[m]["median_seconds_all_queries"]*1000 for m in methods]
            bars = ax.bar(x, values, color=colors, width=.6)
            ax.set_yscale("log")
            ax.bar_label(bars, fmt="%.3f", padding=5)
            positive = [value for value in values if value > 0]
            ax.set(title="Offline search latency · logarithmic axis", ylabel="Median milliseconds (log scale)",
                   ylim=(min(positive)/2, max(positive)*3))
        else:
            if not any(summary[m]["successes"] for m in methods):
                # A flat line at zero hides the result and overlaps the axis.
                statuses = sorted({v.get("status", "unspecified") for row in report["results"]
                                   for v in row["methods"].values() if v["path"] is None})
                counts = {s: [sum(row["methods"][m]["path"] is None and
                                  row["methods"][m].get("status", "unspecified") == s
                                  for row in report["results"]) for m in methods] for s in statuses}
                left = np.zeros(len(methods))
                for status, values in counts.items():
                    bars = ax.barh(x, values, left=left, label=status.replace("_", " "), height=.5)
                    ax.bar_label(bars, labels=[str(v) if v else "" for v in values], label_type="center")
                    left += values
                ax.set(yticks=x, yticklabels=labels, xlabel="Queries that stopped without a route",
                       title="Why search stopped (all queries failed)", xlim=(0, n*1.1))
                ax.legend(frameon=False, fontsize=10, loc="lower right")
                return
            budgets = sorted(set(b for b in [1, 2, 3, 5, 10, 20, 50, 100, report["budget_expansions"]]
                                 if b <= report["budget_expansions"]))
            maximum = 0
            for method in methods:
                values = [row["methods"][method] for row in report["results"]]
                rates = [100*sum(v["path"] is not None and v["expanded"] <= b for v in values)/len(values) for b in budgets]
                maximum = max(maximum, max(rates))
                ax.plot(budgets, rates, marker="o", color=COLORS[method], label=LABELS[method], linewidth=2)
            ax.set(title="Success versus search budget", xlabel="Maximum articles expanded (log scale)",
                   ylabel="Goals reached (%)", xscale="log", ylim=(0, min(105, max(5, maximum*1.2))))
            ax.legend(frameon=False, fontsize=10)
        if index < 3:
            ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for index, ax in enumerate(axes.flat):
        panel(ax, index)
    fig.suptitle(caption, fontsize=15)
    if report.get("seconds_per_query"):
        fig.supxlabel(f"Snapshot {report.get('snapshot')} · cold feature calculation included · fixed time cap at every budget", fontsize=11)
    save(fig, output, "search_comparison")
    # Separate large panels avoid squeezing four charts onto a laptop screen.
    for index, name in enumerate(["search_work", "route_quality", "search_latency", "success_budget"]):
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        panel(ax, index)
        fig.suptitle(caption, fontsize=12)
        save(fig, output, name)
    with (output / "search_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *next(iter(summary.values())).keys()])
        writer.writeheader()
        for method in methods:
            writer.writerow({"method": method, **summary[method]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("models/raw"), help="Raw-model bundles directory")
    parser.add_argument("--output", type=Path, default=Path("figures/raw/baseline"))
    parser.add_argument("--comparison", type=Path, help="Use a different measured comparison report")
    parser.add_argument("--training-metrics", type=Path, nargs="+", help="Plot up to three specified training histories")
    parser.add_argument("--search-only", action="store_true", help="Plot only search results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    if not args.search_only:
        training_curves(args.runs, args.output, args.training_metrics)
        loss_shape(args.output)
    search_plots(read(args.comparison or Path("reviews/results/comparison_1100.json")), args.output)
    print(f"Saved PNG/PDF charts and CSV summary to {args.output}")


if __name__ == "__main__":
    main()
