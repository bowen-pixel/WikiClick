"""Render only measured million-article ranking and branching results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from codes.api.million_semantic_data import OUT
from codes.api.sample_dataset import save_json

NAMES = {"raw":"Raw distance model","title_rank":"Title next-link ranker","semantic_rank":"Text/category ranker",
         "continued":"Ordinary continuation","corrected":"Correction with rollback"}
COLORS = ["#61778e","#4489ae","#138b83","#bc8d35","#834b9d"]


def main():
    data = json.loads((OUT/"results.json").read_text())
    output = Path("figures/next_link/ranking_million");output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        for suffix in ["png","pdf"]:fig.savefig(output/f"{name}.{suffix}",dpi=200,bbox_inches="tight")
        plt.close(fig)
    budgets = data["budgets"]
    means = {}
    for name in NAMES:
        values = [r[name] for r in data["runs"].values()]
        means[name] = {"next_link_accuracy":float(np.mean([r["next_link"]["optimal_next_link_accuracy"] for r in values])),
                       "greedy_success":float(np.mean([r["greedy"]["success_rate"] for r in values])),
                       "branching":{str(b):{key:float(np.mean([r["branching"][str(b)][key] for r in values])) for key in
                                            ["success_rate","optimal_rate","mean_expanded_all","mean_candidates_scored_all","median_ms"]} for b in budgets}}
    save_json(OUT/"aggregate.json",means)
    for metric,heading,filename in [("success_rate","Goals reached","branching_success"),("optimal_rate","Minimum-click routes found","branching_optimal")]:
        fig,ax = plt.subplots(figsize=(10,6),layout="constrained")
        maximum = max(v["branching"][str(b)][metric] for v in means.values() for b in budgets)
        maximum = max(maximum,max(data["bfs"][str(b)][metric] for b in budgets))
        if maximum == 0:
            ax.axis("off")
            rows = [[NAMES[k],*["0 / "+str(data["queries"]) for _ in budgets]] for k in NAMES]
            rows.append(["BFS",*["0 / "+str(data["queries"]) for _ in budgets]])
            table = ax.table(cellText=rows,colLabels=["Model",*[str(b)+" expansions" for b in budgets]],cellLoc="center",bbox=[0,.25,1,.65])
            table.auto_set_font_size(False);table.set_fontsize(10)
            ax.text(.5,.1,"No measured improvement at these limits.\nAll test pairs have independently verified paths.",ha="center",transform=ax.transAxes)
        else:
            for (name,label),color in zip(NAMES.items(),COLORS):
                ax.plot(budgets,[100*means[name]["branching"][str(b)][metric] for b in budgets],"o-",label=label,color=color,linewidth=2)
            ax.plot(budgets,[100*data["bfs"][str(b)][metric] for b in budgets],"s--",label="BFS",color="black")
            ax.set(xscale="log",xlabel="Expansion budget (log scale)",ylabel="Percent of all test queries",ylim=(-1,min(105,max(5,maximum*115))))
            ax.legend(frameon=False,fontsize=10);ax.grid(alpha=.2)
        fig.suptitle(f"{heading}: {data['graph']['nodes']:,} articles",fontsize=15)
        fig.supxlabel(f"Same {data['queries']} queries · {len(data['seeds'])} seeds · {data['seconds_per_query']:g}s/query cap · precomputed features\n"
                      "Overlapping lines can mean validation retained identical weights; inspect the checkpoint table.",fontsize=10)
        save(fig,filename)
    for field,title,filename in [("next_link_accuracy","Optimal-next-link accuracy","next_link_accuracy"),("greedy_success","20-click navigation success","greedy_success")]:
        fig,ax=plt.subplots(figsize=(10,5.5),layout="constrained")
        values=[100*means[n][field] for n in NAMES]
        ax.barh(list(NAMES.values()),values,color=COLORS,height=.6)
        for i,value in enumerate(values):ax.text(103,i,f"{value:.1f}%",va="center")
        ax.invert_yaxis();ax.set(xlim=(0,114),xlabel="Percent (higher is better)",title=title);ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
        save(fig,filename)
    checkpoints = []
    for seed in data["seeds"]:
        folder=OUT/f"seed_{seed}"
        base=torch.load(folder/"semantic_rank.pt",map_location="cpu",weights_only=True)["state_dict"]
        for name in ["continued","corrected"]:
            checkpoint=torch.load(folder/f"{name}.pt",map_location="cpu",weights_only=True)
            change=sum((checkpoint["state_dict"][k]-v).square().sum().item() for k,v in base.items())**.5
            checkpoints.append({"seed":seed,"stage":name,"selected_epoch":checkpoint["selected_epoch"],"weight_change_l2":change})
    save_json(OUT/"selected_checkpoints.json",checkpoints)
    lines=["# Million-article next-link ranking experiment","",f"Completed graph evaluation: {data['graph']['nodes']:,} nodes, {data['graph']['edges']:,} directed links.","",
           data["protocol"],"", "## Feature coverage","",json.dumps(data["features"],indent=2),"",
           "## Matched branching results","", "Percentages are seed means on one shared query set, not independent repeated test sets. All conditions use the same expansion/time caps. Unsuccessful routes count as non-optimal.","",
           "| Model | Budget | Reached | Optimal | Mean expansions | Mean candidates scored | Median ms (seed mean) |","|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        for b in budgets:
            r=means[name]["branching"][str(b)]
            lines.append(f"| {NAMES[name]} | {b:,} | {r['success_rate']*100:.2f}% | {r['optimal_rate']*100:.2f}% | {r['mean_expanded_all']:.1f} | {r['mean_candidates_scored_all']:.1f} | {r['median_ms']:.2f} |")
    for b in budgets:
        r=data["bfs"][str(b)]
        lines.append(f"| BFS | {b:,} | {r['success_rate']*100:.2f}% | {r['optimal_rate']*100:.2f}% | {r['mean_expanded_all']:.1f} | 0 (no neural scoring) | {r['median_ms']:.2f} |")
    lines += ["","## Correction checkpoint selection","", "Epoch 0 means attempted updates were rejected and the previous semantic checkpoint was retained.","",
              "| Seed | Stage | Selected epoch | Selected parameter change (L2) |","|---:|---|---:|---:|"]
    for r in checkpoints:lines.append(f"| {r['seed']} | {r['stage']} | {r['selected_epoch']} | {r['weight_change_l2']:.6f} |")
    lines += ["","## Limits","", "This is sampled supervision on a million-node graph, not supervision of every article or every pair. The semantic representation uses fixed word-hashed lead text and explicit-category pooling, not a pretrained language model or learned hypergraph propagation. Category metadata differs from the small pilot: wikitext-only categories omit inherited memberships and can include hidden/maintenance categories. A graph snapshot is a collection of revisions, not an atomic instant.","",
              "The raw model receives up to 60,000 distance pairs. Rankers have additional training; semantic rankers have an extra residual head. These differences prevent attributing the raw-versus-refined comparison to the loss alone. Ordinary continuation is the equal-update control for mistake resampling. No test labels, test failures or exact distances are input to the search policy. Models are selected by validation loss before test evaluation. No results are extrapolated from the 1,000-article run.","",
              "Branching here means prioritizing the frontier: valid outgoing links and graph edge costs are unchanged. Fewer expansions do not imply better search if the method reaches fewer goals. Only BFS has the shortest-path guarantee. Deadline enforcement is checked between bounded batches; a running tensor operation can slightly overshoot. Identical selected checkpoints reuse clearly marked evaluation records rather than fabricate independent timing runs."]
    Path("reviews/comparisons/MILLION_RANKING_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(f"Saved measured figures to {output}")


if __name__ == "__main__":main()
