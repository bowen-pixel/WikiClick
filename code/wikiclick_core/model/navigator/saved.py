"""Run the original saved 1,000-article rankers without fetching or learning."""
import argparse
from pathlib import Path
import torch
from .offline_search import load_graph
from ..next_link.train import DATA, OUT, navigate
from ..next_link.scoring import Ranker
from ..neural.network import DistanceModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=207)
    parser.add_argument("--model", choices=["raw", "title_rank", "semantic_rank", "continued", "corrected", "evolutionary", "random_nas"], default="corrected")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--search-run", type=Path, default=Path("models/evolution/evolution_100"))
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("Positive click limit required")
    graph, graph_id = load_graph(DATA)
    features = torch.load(OUT/"features.pt", map_location="cpu", weights_only=True)
    if features["graph_id"] != graph_id:
        raise ValueError("Feature and graph identity differ")
    if args.model in ("evolutionary", "random_nas"):
        from ..evolutionary.architecture_search import load_selected
        arm = "evolution" if args.model == "evolutionary" else "random"
        model = load_selected(args.search_run/f"{arm}_selected.pt", graph_id)
    else:
        checkpoint = torch.load(OUT/f"seed_{args.seed}"/f"{args.model}.pt", map_location="cpu", weights_only=True)
        if checkpoint["graph_id"] != graph_id:
            raise ValueError("Checkpoint and graph identity differ")
        if args.model == "raw":
            model = Ranker(checkpoint["state_dict"])
        else:
            model = Ranker(DistanceModel().state_dict(), checkpoint["semantic"])
            model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    names = {graph.nodes[n]["title"].casefold(): n for n in graph}
    source = args.source or input("Source: ").strip()
    target = args.target or input("Target: ").strip()
    if source.casefold() not in names or target.casefold() not in names:
        parser.error("Choose source and target titles from the saved 1,000-article graph")
    a, b = names[source.casefold()], names[target.casefold()]
    if a == b:
        print(f"Already at {graph.nodes[a]['title']}; 0 clicks.")
        return
    index = {n: i for i, n in enumerate(features["page_ids"])}
    row = navigate(graph, [{"source": a, "target": b}], index, model,
                   features["title_vectors"], features["semantic_vectors"], args.max_steps)[0]
    print(" -> ".join(graph.nodes[n]["title"] for n in row["visited_path"]))
    print(f"Status: {row['status']}; clicks: {row['clicks']}")


if __name__ == "__main__":
    main()
