"""Plot actual matched 1000-article results and write an experiment report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from next_link_experiment import DATA, OUT
from sample_dataset import save_json

NAMES = {"raw":"Raw distance model", "title_rank":"Title next-link ranker", "semantic_rank":"Text + hypergraph ranker",
         "continued":"Ordinary continuation", "corrected":"Mistake-focused correction"}
COLORS = ["#61778e", "#4489ae", "#138b83", "#bc8d35", "#834b9d"]


def save(fig, name):
    for extension in ("png","pdf"):
        fig.savefig(OUT/"figures"/f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def paired_gain(result, first, second):
    """Paired cluster bootstrap by destination, averaging seeds within queries.

    This conditional interval describes this fixed graph and destination sample;
    it is not evidence of a general scaling law.
    """
    grouped = {}
    for seed in result["seeds"]:
        folder = OUT/f"seed_{seed}"
        a = json.loads((folder/f"{first}.routes.json").read_text())
        b = json.loads((folder/f"{second}.routes.json").read_text())
        assert [(r["source"],r["target"]) for r in a] == [(r["source"],r["target"]) for r in b]
        for x,y in zip(a,b):
            grouped.setdefault(x["target"], []).append(int(y["path"] is not None)-int(x["path"] is not None))
    totals = np.array([[sum(v),len(v)] for v in grouped.values()])
    rng = np.random.default_rng(844)
    indices = rng.integers(0,len(totals),size=(2000,len(totals)))
    draws = totals[indices].sum(1)
    boot = 100*draws[:,0]/draws[:,1]
    return {"gain_percentage_points":float(100*totals[:,0].sum()/totals[:,1].sum()),
            "conditional_95_percent_interval":np.quantile(boot,[.025,.975]).tolist(), "destination_clusters":len(totals)}


def main():
    result = json.loads((OUT/"results.json").read_text())
    semantic = json.loads((DATA/"semantic_manifest.json").read_text())
    (OUT/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
    keys = list(NAMES)
    aggregate = {}
    for key in keys:
        runs = [r["test"][key] for r in result["runs"].values()]
        aggregate[key] = {"next_link_accuracy":float(np.mean([r["next_link"]["optimal_next_link_accuracy"] for r in runs])),
                          "route_success":float(np.mean([r["navigation"]["success_rate"] for r in runs])),
                          "optimal_route_rate":float(np.mean([r["navigation"]["optimal_rate"] for r in runs])),
                          "median_latency_ms_mean_across_seeds":float(np.mean([r["navigation"]["median_ms"] for r in runs]))}
    gains = {"corrected_vs_raw":paired_gain(result,"raw","corrected"),
             "corrected_vs_continued":paired_gain(result,"continued","corrected"),
             "semantic_vs_title":paired_gain(result,"title_rank","semantic_rank")}
    checkpoints = {}
    for seed in result["seeds"]:
        folder = OUT/f"seed_{seed}"
        base = torch.load(folder/"semantic_rank.pt",map_location="cpu",weights_only=True)["state_dict"]
        checkpoints[str(seed)] = {}
        for key in ["continued","corrected"]:
            checkpoint = torch.load(folder/f"{key}.pt",map_location="cpu",weights_only=True)
            change = sum((checkpoint["state_dict"][k]-v).square().sum().item() for k,v in base.items())**.5
            checkpoints[str(seed)][key] = {"selected_epoch":checkpoint["selected_epoch"],"selected_weight_change_l2":change}
    save_json(OUT/"aggregate.json", {"metrics":aggregate,"paired_route_success_gains":gains,"selected_correction_checkpoints":checkpoints})
    if all(c["corrected"]["selected_epoch"] == 0 for c in checkpoints.values()):
        NAMES["corrected"] = "Correction (prior model retained)"
    for metric, name, heading in [("next_link_accuracy","next_link_accuracy","Does the top-ranked link belong to a shortest path?"),
                                   ("route_success","route_success","Complete journeys on identical held-out queries"),
                                   ("optimal_route_rate","optimal_routes","Journeys that reach the goal with minimum clicks")]:
        fig,ax = plt.subplots(figsize=(10,6),layout="constrained")
        values = [100*aggregate[k][metric] for k in keys]
        bars = ax.barh([NAMES[k] for k in keys],values,color=COLORS,height=.6)
        for i,value in enumerate(values):
            ax.text(103,i,f"{value:.1f}%",va="center")
        for i,k in enumerate(keys):
            seed_values = [100*(r["test"][k]["next_link"]["optimal_next_link_accuracy"] if metric == "next_link_accuracy" else
                          r["test"][k]["navigation"]["success_rate" if metric == "route_success" else "optimal_rate"]) for r in result["runs"].values()]
            ax.scatter(seed_values,i+np.linspace(-.12,.12,len(seed_values)),color="black",s=18,zorder=3)
        ax.set(xlim=(0,113),xlabel="Percent of held-out decisions or routes; higher is better",title=heading)
        ax.invert_yaxis();ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
        fig.supxlabel(f"1,000 articles · {result['decision_counts']['test']} matched test queries · {len(result['seeds'])} seeds\n"
                      "Bars are seed means; dots are individual seeds. Routing: 20 clicks, no revisits.",fontsize=10)
        save(fig,name)

    fig,axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    # Use one declared seed for readable histories, rather than averaging epochs selected differently.
    folder = OUT/f"seed_{result['seeds'][0]}"
    for key,color in [("semantic_rank",COLORS[2]),("continued",COLORS[3]),("corrected",COLORS[4])]:
        h = json.loads((folder/f"{key}.history.json").read_text())
        x = [r["epoch"] for r in h]
        axes[0].plot(x,[r["validation_loss"] for r in h],"o-",label=NAMES[key],color=color,markersize=3)
        axes[1].plot(x,[100*r["validation_optimal_next_link_accuracy"] for r in h],"o-",label=NAMES[key],color=color,markersize=3)
    axes[0].set(title="Validation ranking loss (lower is better)",ylabel="Cross-entropy per decision")
    axes[1].set(title="Validation optimal-next-link accuracy",ylabel="Percent correct")
    for ax in axes:
        ax.set_xlabel("Epoch within each stage (stage starts differ)");ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=9)
    fig.supxlabel(f"Seed {result['seeds'][0]}. Continuation and correction start from the SAME selected semantic checkpoint.\n"
                  "Correction changes the training distribution; training-loss values across those stages are not directly comparable.",fontsize=10)
    save(fig,"validation_curves")

    fig,ax = plt.subplots(figsize=(10,4.5),layout="constrained")
    labels = ["Full ranking pipeline − raw model","Correction − ordinary continuation","Semantic context − title ranking"]
    for i,(key,color) in enumerate(zip(gains,[COLORS[4],COLORS[3],COLORS[2]])):
        g = gains[key];lo,hi = g["conditional_95_percent_interval"]
        ax.hlines(i,lo,hi,color=color,linewidth=4)
        ax.plot(g["gain_percentage_points"],i,"o",color=color,markersize=9)
        ax.annotate(f"{g['gain_percentage_points']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]",(hi,i),xytext=(8,0),textcoords="offset points",va="center",fontsize=10)
    limits = [v for g in gains.values() for v in g["conditional_95_percent_interval"]]
    span = max(1,max(limits)-min(limits))
    ax.set(yticks=range(3),yticklabels=labels,xlabel="Change in route success (percentage points)",
           title="Paired gains: intervals crossing zero are inconclusive",xlim=(min(0,min(limits))-span*.1,max(0,max(limits))+span*.8))
    ax.axvline(0,color="0.5",linestyle="--");ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    fig.supxlabel("95% paired bootstrap intervals, clustered by destination; conditional on this graph and the trained seeds.\n"
                  "Raw versus corrected also changes training duration, features and parameter count.",fontsize=10)
    save(fig,"paired_gains")

    lines = ["# 1,000-article next-link ranking experiment", "", "Measured local experiment; no guaranteed improvement is assumed.", "",
             f"Graph: **{result['nodes']:,} articles, {result['edges']:,} directed links**. Graph ID: `{result['graph_id']}`.",
             "The 1,000 nodes are a deterministic induced subset of the existing 1,100-article collection, not 1,000 new downloads.", "",
             f"Fetched current introduction text for {semantic['intros_present']}/1000 articles. Retained {semantic['category_hyperedges']} category hyperedges, covering {semantic['articles_in_hyperedges']} articles. Text collection is later than the hyperlink collection. Empty introductions use zero content features. Sources/revisions are cached.", "",
             "## Protocol", "", result["protocol"], "",
             f"Destinations: 800 train / 100 validation / 100 test. Decisions: {result['decision_counts']}. Seeds: {result['seeds']}.",
             "Each decision contains all outgoing neighbors and all equally optimal labels. Labels are exact in this saved graph, not the full web. Train/validation/test destinations do not overlap. Article text and category membership across the fixed graph are visible as unsupervised transductive features.", "",
             "Raw baseline: original title-only distance architecture (34,945 parameters), trained from random weights on at most 60,000 unique neighbor/destination pairs from the training decisions, Smooth L1 loss. Test/validation targets are never used in optimizer updates.",
             f"Ranking: {result['epochs_per_stage']} extra epochs from the raw checkpoint. Title ranker retains 34,945 parameters; semantic ranker adds a 16,449-parameter residual head using intro hashes and fixed category-hypergraph pooling (51,394 total). This is not a pretrained semantic encoder or a learned hypergraph encoder.",
             f"Correction: {result['correction_epochs']} extra epochs, training mistakes sampled with weight 4. Ordinary continuation receives the same number of updates from the same semantic checkpoint. Both select checkpoints by validation cross-entropy, including the option of retaining the starting checkpoint. Final test labels are opened only after every seed/condition is trained.", "",
             "## Actual held-out results", "", "Percentages are means across seeds; each seed uses the identical test decisions and journeys.", "",
             "| Model | Optimal next link | Goal reached | Optimal route | Median ms (mean across seeds) |", "|---|---:|---:|---:|---:|"]
    for key in keys:
        a = aggregate[key]
        lines.append(f"| {NAMES[key]} | {100*a['next_link_accuracy']:.2f}% | {100*a['route_success']:.2f}% | {100*a['optimal_route_rate']:.2f}% | {a['median_latency_ms_mean_across_seeds']:.3f} |")
    lines += ["",f"BFS optimal reference: {result['bfs']['successes']}/{result['bfs']['queries']} reached; {result['bfs']['optimal_routes']} optimal. BFS has a different work limit and is not presented as a matched navigation policy.","", "## Paired route-success gains", ""]
    for name,key in zip(labels,gains):
        g = gains[key];lo,hi = g["conditional_95_percent_interval"]
        lines.append(f"- {name}: {g['gain_percentage_points']:+.2f} percentage points; conditional 95% interval [{lo:+.2f}, {hi:+.2f}].")
    lines += ["", "## Was the correction retained?", "", "Epoch 0 means validation retained the original semantic checkpoint; correction updates were attempted but not deployed.", "",
              "| Seed | Continuation selected epoch | Correction selected epoch | Correction selected-weight change (L2) |", "|---:|---:|---:|---:|"]
    for seed,c in checkpoints.items():
        lines.append(f"| {seed} | {c['continued']['selected_epoch']} | {c['corrected']['selected_epoch']} | {c['corrected']['selected_weight_change_l2']:.6f} |")
    lines += ["", "Intervals cluster by destination, averaging seeds within each paired query; they do not establish generalization beyond this graph. We did not retune using test outcomes. Correction need not beat ordinary continuation, and semantic context need not help.", "",
              "## Local deployment", "", "`rank_navigate.py` loads saved weights and cached features locally. It does not fetch live pages, update weights during inference, modify the hyperlink graph, or use true distances to choose moves. Seed 207 is the declared default, not selected using test performance.", "", "```bash", "python rank_navigate.py", "python rank_navigate.py --model raw --source 'Graph theory' --target 'Mathematics'", "```", "",
              "Only the saved 1,000 titles are supported. The corrected checkpoint is an experimental alternative, not automatically a better default. Use `--model continued`, `--model semantic_rank`, or `--model title_rank` to compare.", "",
              "## Reproduce", "", "```bash", "python next_link_experiment.py prepare", "python next_link_experiment.py fetch", "python next_link_experiment.py run", "python plot_next_link.py", "```", "",
              "Existing source caches are reused. Training commands restart rather than resume optimizer sessions. No earlier million-node artifacts or checkpoints were overwritten. Self-correction here means offline supervised hard-example resampling, not autonomous fact verification, RL, Coconut, or test-time training."]
    Path("NEXT_LINK_1000.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"metrics":aggregate,"gains":gains},indent=2))


if __name__ == "__main__":
    main()
