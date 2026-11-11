"""Evaluate a frozen model on fresh sparse data, with exact sampled references."""

import argparse
from collections import deque
import gzip
import heapq
import itertools
import json
from pathlib import Path
import statistics
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import torch

from ml_model import DistanceModel, FEATURE_VERSION, pair_features, title_features
from sample_dataset import save_json


class SearchDeadline(Exception):
    pass


def search_csr(graph, source, target, budget, seconds, scorer_factory=None):
    started = time.perf_counter()
    deadline = started+seconds
    scorer = scorer_factory(deadline) if scorer_factory else None
    parents = {source: None}
    counter = itertools.count()
    frontier = deque([source]) if scorer is None else [(0., next(counter), source)]
    expanded = 0
    found = False
    reason = "frontier_exhausted"
    try:
        while frontier:
            if time.perf_counter() > deadline:
                raise SearchDeadline
            node = frontier.popleft() if scorer is None else heapq.heappop(frontier)[2]
            if node == target:
                found = True
                break
            if expanded >= budget:
                reason = "expansion_budget"
                break
            expanded += 1
            neighbors = [int(n) for n in graph.indices[graph.indptr[node]:graph.indptr[node+1]] if int(n) not in parents]
            if target in neighbors:
                parents[target] = node
                found = True
                break
            priorities = scorer(neighbors, target) if scorer is not None and neighbors else []
            for i, candidate in enumerate(neighbors):
                parents[candidate] = node
                if scorer is None:
                    frontier.append(candidate)
                else:
                    heapq.heappush(frontier, (priorities[i], next(counter), candidate))
    except SearchDeadline:
        reason = "time_limit"
    path = None
    if found:
        path = []
        node = target
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        reason = "found"
    return {"path": path, "clicks": len(path)-1 if path else None, "expanded": expanded,
            "seconds": time.perf_counter()-started, "status": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/fresh_million"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/model_1100/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/fresh_million_comparison.json"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--query-file", type=Path, help="Reuse a predeclared graph-specific held-out query set")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seconds-per-query", type=float, default=10)
    args = parser.parse_args()
    if min(args.queries, args.budget, args.seconds_per_query) <= 0:
        parser.error("Budgets and query count must be positive")
    torch.set_num_threads(2)
    metadata = json.loads((args.data / "metadata.json").read_text())
    graph = sparse.load_npz(args.data / "graph.csr.npz").tocsr()
    reverse = graph.transpose().tocsr()
    ids = np.load(args.data / "page_ids.npy")
    with gzip.open(args.data / "nodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        titles = [json.loads(line)["title"] for line in stream]
    if len(titles) != graph.shape[0] or len(ids) != graph.shape[0]:
        raise ValueError("Sparse graph and node tables differ")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["feature_version"] != FEATURE_VERSION:
        raise ValueError("Incompatible title feature version")
    model = DistanceModel(checkpoint["dimensions"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if checkpoint["graph_id"] == metadata["graph_id"]:
        allowed_ids = set(checkpoint["destination_splits"]["test"])
        scope = "Trained expanded-graph model on held-out destination articles"
    else:
        known = set(checkpoint["training_graph_node_ids"])
        allowed_ids = {int(node) for node in ids if int(node) not in known}
        scope = "Frozen model transfer to destination page IDs outside its original graph"
    eligible = np.flatnonzero(np.array([int(node) in allowed_ids for node in ids], dtype=bool))
    rng = np.random.default_rng(207)
    queries = []
    reference_seconds = 0.
    if args.query_file:
        query_data = json.loads(args.query_file.read_text())
        if query_data["graph_id"] != metadata["graph_id"]:
            raise ValueError("Query file belongs to another graph")
        references = {}
        for row in query_data["queries"][:args.queries]:
            source, goal = int(row["source"]), int(row["target"])
            if not (0 <= source < len(ids) and 0 <= goal < len(ids)):
                raise ValueError("Query index outside graph")
            if int(ids[goal]) not in allowed_ids:
                raise ValueError("Query destination is not held out for this checkpoint")
            if (int(ids[source]), int(ids[goal])) != (row["source_page_id"], row["target_page_id"]):
                raise ValueError("Query page IDs differ from graph")
            if goal not in references:
                started = time.perf_counter()
                references[goal] = shortest_path(reverse, directed=True, indices=goal, unweighted=True)
                reference_seconds += time.perf_counter()-started
            distance = references[goal][source]
            if not np.isfinite(distance) or distance < 2 or int(distance) != row["optimal_clicks"]:
                raise ValueError("Query reference label is invalid")
            queries.append({**row, "source_title": titles[source], "target_title": titles[goal]})
        del references
    else:
        goals = rng.choice(eligible, size=min(len(eligible), args.queries*3), replace=False)
        per_goal = max(1, (args.queries+len(goals)-1)//len(goals)) if len(goals) else 0
        for goal in goals:
            started = time.perf_counter()
            distances = shortest_path(reverse, directed=True, indices=int(goal), unweighted=True)
            reference_seconds += time.perf_counter()-started
            sources = np.flatnonzero(np.isfinite(distances) & (distances >= 2))
            for source in rng.choice(sources, size=min(per_goal, len(sources)), replace=False):
                source = int(source)
                queries.append({"source": source, "target": int(goal), "optimal_clicks": int(distances[source]),
                                "source_page_id": int(ids[source]), "target_page_id": int(ids[goal]),
                                "source_title": titles[source], "target_title": titles[int(goal)]})
                if len(queries) == args.queries:
                    break
            if len(queries) == args.queries:
                break
    if len(queries) != args.queries:
        raise ValueError("Not enough eligible graph-reachable queries")

    def factory(neural):
        def build(deadline):
            cache = {}  # Equivalent cold title-feature caches per method/query.
            def vector(node):
                if time.perf_counter() > deadline:
                    raise SearchDeadline
                if node not in cache:
                    cache[node] = title_features(titles[node], checkpoint["dimensions"])
                return cache[node]
            @torch.inference_mode()
            def score(nodes, goal):
                destination = vector(goal)
                result = []
                for start in range(0, len(nodes), 256):
                    batch = nodes[start:start+256]
                    candidates = torch.stack([vector(node) for node in batch])
                    if neural:
                        goal_vectors = destination.expand_as(candidates)
                        result.extend(model(pair_features(candidates, goal_vectors)).tolist())
                    else:
                        result.extend((-(candidates*destination).sum(dim=1)).tolist())
                return result
            return score
        return build
    with torch.inference_mode():
        model(torch.zeros(1, checkpoint["dimensions"]*4))
    results = []
    for i, query in enumerate(queries):
        row = dict(query)
        row["methods"] = {}
        for name, scorer in (("bfs", None), ("text", factory(False)), ("neural", factory(True))):
            result = search_csr(graph, query["source"], query["target"], args.budget, args.seconds_per_query, scorer)
            result["extra_clicks"] = result["clicks"]-query["optimal_clicks"] if result["path"] else None
            if result["path"]:
                for a, b in zip(result["path"], result["path"][1:]):
                    if not graph[a, b]:
                        raise ValueError("Invalid search path")
            row["methods"][name] = result
        results.append(row)
        print(f"Search query {i+1}/{len(queries)} complete", flush=True)
    summary = {}
    for name in ("bfs", "text", "neural"):
        values = [row["methods"][name] for row in results]
        successes = [value for value in values if value["path"]]
        summary[name] = {"queries": len(values), "successes": len(successes), "success_rate": len(successes)/len(values),
                         "optimal_route_rate_all_queries": sum(v["extra_clicks"] == 0 for v in values)/len(values),
                         "mean_extra_clicks_successes_only": statistics.mean(v["extra_clicks"] for v in successes) if successes else None,
                         "mean_expanded_all_queries": statistics.mean(v["expanded"] for v in values),
                         "median_seconds_all_queries": statistics.median(v["seconds"] for v in values)}
    report = {"graph_id": metadata["graph_id"], "graph_nodes": graph.shape[0], "graph_edges": int(graph.nnz),
              "snapshot": metadata["snapshot"], "checkpoint_graph_id": checkpoint["graph_id"],
              "scope": scope, "query_file": str(args.query_file) if args.query_file else None,
              "budget_expansions": args.budget, "seconds_per_query": args.seconds_per_query,
              "reference_seconds": reference_seconds, "summary": summary, "results": results,
              "notes": ["Exact reference distances only for sampled queries, not all pairs.",
                        "Cold per-query feature costs included for both guided methods; no live API calls.",
                        "Both time and expansion budgets apply; budget curves must be interpreted with the fixed time limit."]}
    save_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
