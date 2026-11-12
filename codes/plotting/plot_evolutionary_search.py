"""Render actual NAS measurements, architecture diagrams, and a search animation."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wikiclick-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
import numpy as np


def save(fig, folder, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("models/evolution/evolution_100"))
    parser.add_argument("--output", type=Path, help="Figure directory (defaults to figures/evolution/RUN_NAME)")
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    folder = args.output or Path("figures/evolution") / args.run.name
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                        "axes.spines.right": False, "figure.facecolor": "#f8fafc"})
    colors = {"baseline": "#64748b", "evolution": "#0d9488", "random": "#a855f7"}
    names = {"baseline": "Existing ranker", "evolution": "Evolution selected", "random": "Random-search selected"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    baseline = result["baseline"]["validation_routes"]
    count = result["population"] * result["generations"]
    for arm, search in result["search"].items():
        candidates = search["candidates"]
        for ax, metric, title in [(axes[0, 0], "successes", "Validation: goals reached"),
                                  (axes[0, 1], "optimal_routes", "Validation: shortest routes")]:
            # Plot the actual lexicographically selected incumbent, not a fictional surface.
            from codes.models.evolution.evolutionary_search import fitness
            best, values = result["baseline"], [baseline[metric]]
            for candidate in candidates:
                if fitness(candidate) > fitness(best):
                    best = candidate
                values.append(best["validation_routes"][metric])
            ax.step(range(len(values)), values, where="post", label=names[arm], color=colors[arm], linewidth=2,
                    linestyle="--" if arm == "random" else "-")
            ax.set(title=title, xlabel="Candidate architectures evaluated", ylabel=f"Journeys out of {result['samples']}",
                   xlim=(0, count))
            ax.grid(alpha=.2)
            ax.set_ylim((90, 102) if metric == "successes" else (60, 80))
        selected = search["selected"]
        if selected["config"] is not None:
            history = selected["history"]
            axes[1, 0].plot([h["epoch"] for h in history], [h["validation_rank"]["loss"] for h in history],
                            "o-", label=f"{arm}: validation", color=colors[arm])
            axes[1, 0].plot([h["epoch"] for h in history if h["train_loss"] is not None],
                            [h["train_loss"] for h in history if h["train_loss"] is not None],
                            "--", label=f"{arm}: training", color=colors[arm], alpha=.6)
            chosen = history[selected["selected_epoch"]]
            axes[1, 0].scatter(chosen["epoch"], chosen["validation_rank"]["loss"], marker="*",
                               s=160, color=colors[arm], edgecolor="black", zorder=5)
    axes[0, 0].legend(fontsize=9, frameon=False)
    axes[1, 0].set(title="Selected candidates: measured learning curves", xlabel="Epoch", ylabel="Ranking cross-entropy")
    axes[1, 0].grid(alpha=.2)
    axes[1, 0].legend(fontsize=9, frameon=False)
    x = np.arange(3)
    for offset, metric, label, alpha in [(-.18, "successes", "Reached goal", 1), (.18, "optimal_routes", "Shortest route", .45)]:
        values = [result["test"][k][metric] for k in names]
        bars = axes[1, 1].bar(x + offset, values, width=.34, color=list(colors.values()), alpha=alpha, label=label)
        axes[1, 1].bar_label(bars, padding=3, fontsize=10)
    axes[1, 1].set(xticks=x, xticklabels=["Existing", "Evolution", "Random search"],
                   ylim=(0, result["samples"] * 1.25), title="Final test: 100 matched journeys",
                   ylabel=f"Journeys out of {result['samples']}")
    axes[1, 1].legend(fontsize=9, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("WikiClick | evolutionary architecture-search pilot", fontsize=19, weight="bold")
    fig.supxlabel(f"1,000-article fixed graph · {count} candidates per search · {result['epochs_per_candidate']} epochs each · one seed\n"
                  "Stars mark selected epochs; search curves overlap. Test pairs are new; destinations were evaluated previously.", fontsize=10)
    save(fig, folder, "search_dashboard")

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 5.2))
    ax.axis("off")
    selected = result["search"]["evolution"]["selected"]
    config = selected["config"]

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.12", facecolor=color, edgecolor="#cbd5e1"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#475569"})

    box(.2, 1.8, 2.1, 1.3, "Candidate link\n+ destination\ncached features", "#e2e8f0")
    box(3.2, 3.2, 3.8, 1.1, "Existing trained ranker\n51,394 frozen parameters", "#e2e8f0")
    detail = (f"Evolved residual branch\n{config['features']} features · {config['depth']} hidden layer(s)\n"
              f"width {config['width']} · {config['activation'].upper()}" if config else "Validation retained baseline\nNo residual branch deployed")
    box(3.2, .6, 3.8, 1.4, detail, "#ccfbf1")
    box(8.0, 1.9, 1.1, 1.1, "Add\nscores", "#fef3c7")
    box(10.0, 1.9, 1.6, 1.1, "Rank valid\noutgoing links", "#dbeafe")
    arrow((2.4, 2.8), (3.1, 3.7))
    arrow((2.4, 2.1), (3.1, 1.3))
    arrow((7.1, 3.7), (8, 2.8))
    arrow((7.1, 1.3), (8, 2.1))
    arrow((9.2, 2.45), (9.9, 2.45))
    ax.set_title("Selected architecture: preserve the ranker, evolve a refinement branch", fontsize=17, pad=20)
    fig.supxlabel("Branches start with zero output; the base score is preserved at initialization.\n"
                  "Search changes branch architecture; gradient training learns its weights. No updates during navigation.", fontsize=11)
    save(fig, folder, "selected_architecture")

    # Discrete architecture map, with no invented interpolation or 3D loss surface.
    candidates = result["search"]["evolution"]["candidates"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    width_positions = {16: 0, 32: 1, 64: 2}
    feature_positions = {"title": 0, "semantic": 1, "both": 2}
    positions = {}
    for c in candidates:
        config = c["config"]
        positions[c["id"]] = (width_positions[config["width"]] + (-.10 if config["activation"] == "relu" else .10),
                              feature_positions[config["features"]] + (-.12 if config["depth"] == 1 else .12))
    minimum = min(c["validation_routes"]["optimal_routes"] for c in candidates)
    maximum = max(c["validation_routes"]["optimal_routes"] for c in candidates)
    norm = plt.Normalize(minimum - .5, maximum + .5)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=left, label="Validation shortest routes")

    def frame(number):
        left.clear()
        right.clear()
        visible = candidates[:number + 1]
        for c in visible:
            pos = positions[c["id"]]
            if c["parent"]:
                parent = positions[c["parent"]]
                left.annotate("", xy=pos, xytext=parent, arrowprops={"arrowstyle": "->", "color": "#94a3b8", "alpha": .7,
                                                                   "shrinkA": 12, "shrinkB": 12})
            left.scatter(*pos, c=[c["validation_routes"]["optimal_routes"]], norm=norm, cmap="viridis", s=220,
                         marker="o" if c["config"]["depth"] == 1 else "s", edgecolor="white", zorder=3)
            left.annotate(c["id"].split("_")[-1], pos, ha="center", va="center", fontsize=8, color="white")
        left.set(xticks=[0, 1, 2], xticklabels=[16, 32, 64], yticks=[0, 1, 2],
                 yticklabels=["Title", "Text/category", "Both"], xlim=(-.4, 2.4), ylim=(-.4, 2.4),
                 xlabel="Branch width", ylabel="Branch inputs", title=f"Evaluated architecture {number + 1}/{len(candidates)}")
        from codes.models.evolution.evolutionary_search import fitness
        best, optimal, success = result["baseline"], [], []
        for c in visible:
            if fitness(c) > fitness(best):
                best = c
            optimal.append(best["validation_routes"]["optimal_routes"])
            success.append(best["validation_routes"]["successes"])
        right.step(range(1, len(visible) + 1), optimal, label="Shortest routes", color="#0d9488", where="post")
        right.step(range(1, len(visible) + 1), success, label="Goals reached", color="#2563eb", where="post")
        right.set(xlim=(.5, len(candidates) + .5), ylim=(0, result["samples"] + 5), xlabel="Candidates evaluated",
                  ylabel=f"Validation journeys out of {result['samples']}", title="Best selected model so far")
        right.grid(alpha=.2)
        right.legend(loc="lower right")
        fig.suptitle("Evolution on a discrete architecture space | measured validation results", fontsize=15)
        fig.supxlabel("Arrows show parent mutations. Circles: 1 hidden layer; squares: 2. Horizontal offset: activation.\n"
                      "Overlapping points can represent repeated coordinates. This is not a continuous loss landscape.", fontsize=9)

    animation = FuncAnimation(fig, frame, frames=len(candidates), interval=700)
    animation.save(folder / "evolution.gif", writer=PillowWriter(fps=1.5), dpi=100)
    frame(len(candidates) - 1)
    save(fig, folder, "architecture_search_map")
    print(folder.resolve())


if __name__ == "__main__":
    main()
