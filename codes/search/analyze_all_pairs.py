"""Compute exact directed shortest-distance statistics for every distinct pair."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import networkx as nx


def analyze(graph, distance_writer=None):
    """One forward BFS per source; exclude the zero-distance self pairs."""
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected a simple directed graph")
    if any(data.get("weight", 1) != 1 for _, _, data in graph.edges(data=True)):
        raise ValueError("BFS requires unit-weight edges for this experiment")
    nodes = sorted(graph)
    histogram = Counter()
    sources = []
    for source in nodes:
        distances = nx.single_source_shortest_path_length(graph, source)
        values = [distance for target, distance in distances.items() if target != source]
        histogram.update(values)
        sources.append({
            "source_id": source, "title": graph.nodes[source].get("title", str(source)),
            "reachable_targets": len(values), "unreachable_targets": len(nodes)-1-len(values),
            "mean_optimal_clicks_reachable": sum(values)/len(values) if values else None,
            "max_finite_distance": max(values) if values else None,
        })
        if distance_writer is not None:
            for target in nodes:
                if source != target:
                    distance_writer.writerow((source, target, distances.get(target)))
    reachable = sum(histogram.values())
    total = len(nodes)*(len(nodes)-1)
    distance_sum = sum(distance*count for distance, count in histogram.items())
    def quantile(q):
        if not reachable:
            return None
        # Nearest-rank percentile; for median average the two central ranks.
        import math
        rank = max(1, math.ceil(q*reachable))
        cumulative = 0
        for distance, count in sorted(histogram.items()):
            cumulative += count
            if cumulative >= rank:
                return distance
    def rank_value(rank):
        cumulative = 0
        for distance, count in sorted(histogram.items()):
            cumulative += count
            if cumulative >= rank:
                return distance
    median = ((rank_value((reachable+1)//2)+rank_value((reachable+2)//2))/2
              if reachable else None)
    report = {
        "articles": len(nodes), "edges": graph.number_of_edges(),
        "pair_definition": "Ordered source-to-target pairs; distinct endpoints only",
        "total_ordered_pairs": total, "reachable_pairs": reachable,
        "unreachable_pairs": total-reachable,
        "reachable_fraction": reachable/total if total else None,
        "sum_optimal_clicks_reachable": distance_sum,
        "mean_optimal_clicks_reachable": distance_sum/reachable if reachable else None,
        "median_optimal_clicks_reachable": median,
        "p95_optimal_clicks_reachable": quantile(.95),
        "maximum_finite_distance": max(histogram) if histogram else None,
        "distance_histogram": dict(sorted(histogram.items())),
        "self_pairs_excluded": len(nodes),
        "unreachable_policy": "No finite shortest distance; excluded from mean, not assigned zero",
        "method": "Exact NetworkX unweighted BFS from every source; no neural predictions",
        "scope": "Distances apply only to the saved directed graph, not all Wikipedia",
    }
    return report, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/sample_1100/graph.json"))
    parser.add_argument("--output", type=Path, default=Path("reviews/results/all_pairs_1100"))
    parser.add_argument("--save-distances", action="store_true", help="Also save one CSV row per ordered pair, gzip-compressed")
    args = parser.parse_args()
    frozen = json.loads(args.data.read_text(encoding="utf-8"))
    content = {key: frozen[key] for key in ("nodes", "edges")}
    graph_id = "sha256:" + hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if graph_id != frozen["graph_id"]:
        parser.error("Frozen graph content does not match its graph ID")
    graph = nx.DiGraph()
    for node in frozen["nodes"]:
        graph.add_node(node["pageid"], title=node["title"])
    for edge in frozen["edges"]:
        graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.save_distances:
        with gzip.open(args.output / "pair_distances.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source_id", "target_id", "optimal_clicks"))
            report, sources = analyze(graph, writer)
    else:
        report, sources = analyze(graph)
    report.update({"graph_id": graph_id, "analysis_seconds": time.perf_counter()-started,
                   "pair_distances_file": "pair_distances.csv.gz" if args.save_distances else None,
                   "empty_csv_distance_means": "No route in this saved graph"})
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    with (args.output / "per_source.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "title", "reachable_targets", "unreachable_targets", "mean_optimal_clicks_reachable", "max_finite_distance"])
        writer.writeheader()
        writer.writerows(sources)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
