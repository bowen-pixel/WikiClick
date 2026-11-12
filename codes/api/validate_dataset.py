"""Independently verify frozen graph structure, label files, and split boundaries."""

import argparse
import json
from pathlib import Path

import networkx as nx

from codes.api.sample_dataset import freeze_graph, save_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(folder, retained_from=None):
    dataset = json.loads((folder / "articles.json").read_text())
    frozen = json.loads((folder / "graph.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    graph, rebuilt = freeze_graph(dataset)
    require(rebuilt == frozen, "Frozen graph differs from canonical article records")
    require(frozen["graph_id"] == manifest["graph_id"], "Manifest graph ID differs")
    require(len(dataset["pages"]) == graph.number_of_nodes() == manifest["articles"], "Duplicate or missing source articles")
    require(all(page.get("links_complete") for page in dataset["pages"]), "Incomplete outgoing-link collection")
    require(graph.number_of_edges() == manifest["edges"], "Incorrect edge count")
    require(nx.number_of_selfloops(graph) == 0, "Unexpected self-loop")
    require(all(edge["weight"] == 1 for _, _, edge in graph.edges(data=True)), "Unexpected edge weight")
    retained = None
    if retained_from is not None:
        original = json.loads((retained_from / "graph.json").read_text())
        old_ids = {node["pageid"] for node in original["nodes"]}
        require(old_ids <= set(graph), "Original articles are missing from expansion")
        retained = len(old_ids)
    # Forward BFS per candidate independently checks the reverse-BFS export.
    reference = {}
    used_goals = set()
    total = 0
    for split, info in manifest["training"]["splits"].items():
        goals = set(info["destination_ids"])
        require(not goals & used_goals, "Destination partitions overlap")
        used_goals.update(goals)
        count = nulls = 0
        seen = set()
        with (folder / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                a, b, d = (row[key]["pageid"] for key in ("current_article", "candidate_next_article", "destination"))
                require(row["graph_id"] == frozen["graph_id"] and row["split"] == split, "Row provenance mismatch")
                require(d in goals and a != d and graph.has_edge(a, b), "Invalid candidate or destination")
                require((a, b, d) not in seen, "Duplicate training row")
                seen.add((a, b, d))
                if b not in reference:
                    reference[b] = dict(nx.single_source_shortest_path_length(graph, b))
                distance = reference[b].get(d)
                require(row["label_distance"] == distance, "Incorrect shortest-distance label")
                require(row["reachable_in_graph"] == (distance is not None), "Incorrect reachability label")
                count += 1
                nulls += distance is None
        require(count == info["rows"] and nulls == info["unreachable_rows"], "Manifest row counts differ")
        total += count
    require(used_goals == set(graph), "Destination split coverage differs from graph")
    report = {"graph_id": frozen["graph_id"], "articles": len(graph), "edges": graph.number_of_edges(),
              "training_rows_checked": total, "all_labels_match_networkx": True,
              "destination_splits_disjoint": True, "original_articles_retained": retained,
              "note": "Structural and label verification, not Wikipedia-wide coverage or a content audit."}
    save_json(folder / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/sample_100"))
    parser.add_argument("--retained-from", type=Path)
    args = parser.parse_args()
    validate(args.data, args.retained_from)
