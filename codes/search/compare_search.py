"""Compare BFS, title similarity, and neural greedy best-first on saved graphs."""

import argparse
from collections import deque
import heapq
import itertools
import json
from pathlib import Path
import random
import statistics
import time

import networkx as nx
import torch

from codes.models.raw.ml_model import DistanceModel, FEATURE_VERSION, pair_features, title_features
from codes.api.sample_dataset import save_json


def reconstruct(parents, target):
    path = []
    while target is not None:
        path.append(target)
        target = parents[target]
    return path[::-1]


def search(graph, source, target, budget, score=None):
    """BFS if score is None; otherwise greedy best-first (not guaranteed optimal)."""
    counter = itertools.count()
    parents = {source: None}
    queue = deque([source]) if score is None else [(0.0, next(counter), source)]
    expanded = 0
    started = time.perf_counter()
    path = None
    while queue:
        node = queue.popleft() if score is None else heapq.heappop(queue)[2]
        if node == target:
            path = reconstruct(parents, target)
            break
        if expanded >= budget:
            break
        expanded += 1
        neighbors = [candidate for candidate in sorted(graph.successors(node)) if candidate not in parents]
        # All methods stop as soon as a goal link is observed. For BFS this
        # remains optimal and avoids inflating its expansion count.
        if target in neighbors:
            parents[target] = node
            path = reconstruct(parents, target)
            break
        priorities = score(neighbors, target) if score is not None and neighbors else []
        for index, candidate in enumerate(neighbors):
            parents[candidate] = node
            if score is None:
                queue.append(candidate)
            else:
                heapq.heappush(queue, (priorities[index], next(counter), candidate))
    return {"path": path, "expanded": expanded, "seconds": time.perf_counter()-started,
            "clicks": len(path)-1 if path is not None else None}


def load_graph(folder):
    frozen = json.loads((folder / "graph.json").read_text())
    graph = nx.DiGraph()
    for node in frozen["nodes"]:
        graph.add_node(node["pageid"], title=node["title"])
    for edge in frozen["edges"]:
        graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    return graph, frozen["graph_id"]


def choose_queries(graph, destinations, count, seed):
    """Reference distances select fixed reachable queries, never guide searches."""
    goals = sorted(destinations)
    rng = random.Random(seed)
    rng.shuffle(goals)
    queries = []
    for target in goals:
        distances = nx.single_source_shortest_path_length(graph.reverse(copy=False), target)
        sources = sorted(source for source, distance in distances.items() if distance >= 2)
        rng.shuffle(sources)
        # Several sources per goal while retaining a variety of held-out goals.
        for source in sources[:max(1, (count+len(goals)-1)//len(goals))]:
            queries.append({"source": source, "target": target, "optimal_clicks": distances[source]})
        if len(queries) >= count:
            break
    if not queries:
        raise ValueError("No eligible reachable queries of at least two clicks")
    return queries[:count]


def benchmark(data, checkpoint_path, output, queries=100, budget=200, novel=False):
    torch.set_num_threads(2)
    graph, graph_id = load_graph(data)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["feature_version"] != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version is incompatible")
    model = DistanceModel(checkpoint["dimensions"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    known = set(checkpoint["training_graph_node_ids"])
    if novel:
        goals = set(graph) - known
        scope = "New destination articles outside the model's original graph"
    elif graph_id == checkpoint["graph_id"]:
        goals = set(checkpoint["destination_splits"]["test"])
        scope = "Held-out test destinations on the model's shared graph"
    else:
        raise ValueError("For a different graph use --novel-destinations to exclude original article destinations")
    evaluation_queries = choose_queries(graph, goals, queries, checkpoint["seed"])
    started = time.perf_counter()
    vectors = {node: title_features(graph.nodes[node]["title"], checkpoint["dimensions"]) for node in graph}
    preprocessing = time.perf_counter()-started

    def text_score(nodes, goal):
        return [-(vectors[node] * vectors[goal]).sum().item() if node != goal else -2.0 for node in nodes]

    @torch.inference_mode()
    def neural_score(nodes, goal):
        values = torch.stack([pair_features(vectors[node], vectors[goal]) for node in nodes])
        scores = model(values).tolist()
        return [0.0 if node == goal else value for node, value in zip(nodes, scores)]

    # Warm up inference outside query timings; no labels or updates involved.
    neural_score([next(iter(graph))], next(iter(graph)))
    results = []
    for query in evaluation_queries:
        row = dict(query)
        row["methods"] = {}
        for method, scorer in (("bfs", None), ("text", text_score), ("neural", neural_score)):
            result = search(graph, query["source"], query["target"], budget, scorer)
            if result["path"] is not None:
                if not all(graph.has_edge(a, b) for a, b in zip(result["path"], result["path"][1:])):
                    raise ValueError("Search returned an invalid path")
                result["extra_clicks"] = result["clicks"] - query["optimal_clicks"]
            else:
                result["extra_clicks"] = None
            row["methods"][method] = result
        results.append(row)
    summary = {}
    for method in ("bfs", "text", "neural"):
        values = [row["methods"][method] for row in results]
        successes = [value for value in values if value["path"] is not None]
        summary[method] = {
            "queries": len(values), "successes": len(successes), "success_rate": len(successes)/len(values),
            "optimal_route_rate_all_queries": sum(v["extra_clicks"] == 0 for v in values)/len(values),
            "mean_extra_clicks_successes_only": statistics.mean(v["extra_clicks"] for v in successes) if successes else None,
            "mean_expanded_all_queries": statistics.mean(v["expanded"] for v in values),
            "median_seconds_all_queries": statistics.median(v["seconds"] for v in values),
        }
    report = {"graph_id": graph_id, "checkpoint_graph_id": checkpoint["graph_id"],
              "scope": scope, "budget_expansions": budget,
              "feature_preprocessing_seconds": preprocessing, "summary": summary, "results": results,
              "notes": ["Offline searches: zero API calls; no live fetching speed claim.",
                        "Only graph-reachable queries of at least two clicks; failures under the budget are retained.",
                        "Shared title preprocessing reported separately; per-query neural inference is timed.",
                        "Greedy best-first scores do not guarantee minimum clicks."]}
    save_json(output, report)
    print(json.dumps(summary, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/sample_100"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/raw/model_100/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("reviews/results/comparison_100.json"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--novel-destinations", action="store_true")
    args = parser.parse_args()
    if args.queries < 1 or args.budget < 1:
        parser.error("queries and budget must be positive")
    benchmark(args.data, args.checkpoint, args.output, args.queries, args.budget, args.novel_destinations)


if __name__ == "__main__":
    main()
