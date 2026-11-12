"""Run a saved local next-link policy. No network access or test-time learning."""
import argparse
import json
from pathlib import Path

import torch

from codes.search.compare_search import load_graph
from codes.models.next_link.next_link_experiment import DATA, OUT, Ranker, navigate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=207)
    parser.add_argument("--model", choices=["raw","title_rank","semantic_rank","continued","corrected","evolutionary","random_nas"], default="corrected")
    parser.add_argument("--search-run", type=Path, default=Path("models/evolution/evolution_100"))
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    graph, graph_id = load_graph(DATA)
    features = torch.load(OUT/"features.pt", map_location="cpu", weights_only=True)
    if features["graph_id"] != graph_id:
        raise ValueError("Checkpoint, features and graph differ")
    if args.model in ("evolutionary", "random_nas"):
        from codes.models.evolution.evolutionary_search import load_selected
        arm = "evolution" if args.model == "evolutionary" else "random"
        model = load_selected(args.search_run/f"{arm}_selected.pt", graph_id)
    else:
        checkpoint = torch.load(OUT/f"seed_{args.seed}"/f"{args.model}.pt", map_location="cpu", weights_only=True)
        if checkpoint["graph_id"] != graph_id:
            raise ValueError("Checkpoint, features and graph differ")
        if args.model == "raw":
            model = Ranker(checkpoint["state_dict"])
        else:
            # Construct matching architecture, then load the complete saved ranker.
            from codes.models.raw.ml_model import DistanceModel
            model = Ranker(DistanceModel().state_dict(), checkpoint["semantic"])
            model.load_state_dict(checkpoint["state_dict"])
    names = {graph.nodes[i]["title"].casefold(): i for i in graph}
    source = args.source or input("Source article: ").strip()
    target = args.target or input("Destination article: ").strip()
    if source.casefold() not in names or target.casefold() not in names:
        raise SystemExit("Source and destination must be titles in data/graphs/ranking_1000/graph.json")
    a,d = names[source.casefold()], names[target.casefold()]
    if a == d:
        print(f"Already at {graph.nodes[d]['title']}; 0 clicks.")
        return
    index = {pageid:i for i,pageid in enumerate(features["page_ids"])}
    row = navigate(graph,[{"source":a,"target":d}],index,model,features["title_vectors"],features["semantic_vectors"],args.max_steps)[0]
    print(" -> ".join(graph.nodes[i]["title"] for i in row["visited_path"]))
    print(f"Model: {args.model}; status: {row['status']}; clicks: {row['clicks']}; {row['seconds']*1000:.2f} ms")


if __name__ == "__main__":
    main()
