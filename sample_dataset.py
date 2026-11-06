"""Collect a resumable 100-article graph and export graph-relative training rows."""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import httpx
import networkx as nx

from api_endpoint import USER_AGENT, create_client, get_resolved_page_links
from graph_demo import build_graph, title_key

SEEDS = [
    "Graph theory", "Computer science", "Mathematics", "Physics", "Biology",
    "History", "Geography", "Art", "Canada", "Music",
]
DEFAULT_OUTPUT = Path("data/graphs/sample_100")


def save_json(path, value):
    """Replace a JSON file atomically so interrupted writes do not corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_sample(fetch, cache_dir, count=100, seeds=SEEDS, random_seed=207):
    """Multi-source BFS, shuffling each source's frontier deterministically.

    Only successful full article records count toward the requested total.
    Replaying the same frontier from cached records makes interrupted runs resumable.
    This is a topical, linked sample, not uniform sampling of Wikipedia.
    """
    if not 2 <= count <= 1000:
        raise ValueError("Article count must be between 2 and 1000.")
    queue = deque(seeds)
    queued = {title_key(title) for title in seeds}
    selected = {}
    failures = []
    attempts = 0
    while queue and len(selected) < count and attempts < count * 3:
        title = queue.popleft()
        attempts += 1
        cache_file = cache_dir / (hashlib.sha256(title_key(title).encode()).hexdigest() + ".json")
        if cache_file.exists():
            record = json.loads(cache_file.read_text(encoding="utf-8"))
            if not record.get("links_complete"):
                raise ValueError(f"Incomplete cache record: {cache_file}")
        else:
            try:
                record = fetch(title)
            except ValueError as error:
                failures.append({"title": title, "error": str(error)})
                continue
            save_json(cache_file, record)
        if record["pageid"] in selected:
            continue
        selected[record["pageid"]] = record
        print(f"[{len(selected)}/{count}] {record['title']}", flush=True)
        candidates = sorted(record["link_targets"], key=lambda page: page["pageid"])
        random.Random(f"{random_seed}:{record['pageid']}").shuffle(candidates)
        for candidate in candidates:
            key = title_key(candidate["title"])
            if key not in queued:
                queue.append(candidate["title"])
                queued.add(key)
    if len(selected) != count:
        raise RuntimeError(f"Collected only {len(selected)}/{count} articles; cache retained for resuming.")
    return {
        "schema_version": 2, "language": "en", "pages": list(selected.values()),
        "endpoint": "https://en.wikipedia.org/w/api.php", "user_agent": USER_AGENT,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sampling": {"method": "multi-source BFS with shuffled neighbors", "seeds": list(seeds),
                     "random_seed": random_seed, "requested_articles": count,
                     "attempted_sources": attempts, "failures": failures},
        "scope": "Selected fully fetched article sources; canonical existing namespace-0 targets. API links may include templates.",
        "snapshot_caveat": "Live API responses collected at different times; not an atomic Wikipedia revision snapshot.",
    }


def freeze_graph(dataset):
    graph, _ = build_graph(dataset)
    nodes = [{"pageid": node, **graph.nodes[node]} for node in sorted(graph)]
    edges = [{"source": source, "target": target, "weight": 1}
             for source, target in sorted(graph.edges)]
    content = {"nodes": nodes, "edges": edges}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    graph_id = "sha256:" + digest
    dataset["graph_id"] = graph_id
    frozen = {"schema_version": 1, "graph_id": graph_id, "directed": True,
              "scope": "Shortest distances apply only to this saved graph.", **content}
    return graph, frozen


def export_training(graph, graph_id, output, random_seed=207):
    """Split by destination; null means unreachable within the saved graph."""
    destinations = sorted(graph)
    random.Random(random_seed).shuffle(destinations)
    n = len(destinations)
    train_end, validation_end = int(n * .8), int(n * .9)
    partitions = {"train": destinations[:train_end],
                  "validation": destinations[train_end:validation_end],
                  "test": destinations[validation_end:]}
    counts = {}
    sample = None
    for split, goals in partitions.items():
        path = output / f"{split}.jsonl"
        rows = 0
        unreachable = 0
        with path.open("w", encoding="utf-8") as stream:
            for destination in goals:
                distances = nx.single_source_shortest_path_length(graph.reverse(copy=False), destination)
                for current in sorted(graph):
                    if current == destination:  # Terminal states do not need a next click.
                        continue
                    for candidate in sorted(graph.successors(current)):
                        distance = distances.get(candidate)
                        row = {
                            "graph_id": graph_id, "split": split,
                            "current_article": {"pageid": current, "title": graph.nodes[current]["title"]},
                            "destination": {"pageid": destination, "title": graph.nodes[destination]["title"]},
                            "candidate_next_article": {"pageid": candidate, "title": graph.nodes[candidate]["title"]},
                            "label_distance": distance,
                            "reachable_in_graph": distance is not None,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows += 1
                        unreachable += distance is None
                        if sample is None and distance is not None and candidate != destination:
                            sample = row
        counts[split] = {"rows": rows, "unreachable_rows": unreachable, "destination_ids": goals}
    return {"graph_id": graph_id, "random_seed": random_seed,
            "split_rule": "Disjoint destination articles, 80/10/10; shared fixed graph, transductive evaluation.",
            "label_definition": "Minimum edge count from candidate_next_article to destination in this graph; excludes current-to-candidate click.",
            "null_policy": "No route within this graph. Do not train numeric regression directly on null labels.",
            "splits": counts, "example": sample}


def write_outputs(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    graph, frozen = freeze_graph(dataset)
    save_json(output / "articles.json", dataset)
    save_json(output / "graph.json", frozen)
    training = export_training(graph, frozen["graph_id"], output)
    stats = {
        "graph_id": frozen["graph_id"], "articles": graph.number_of_nodes(),
        "edges": graph.number_of_edges(), "all_edge_weights_one": all(d["weight"] == 1 for _, _, d in graph.edges(data=True)),
        "weak_components": nx.number_weakly_connected_components(graph),
        "strong_components": nx.number_strongly_connected_components(graph),
        "outside_target_references": sum(t["pageid"] not in graph for p in dataset["pages"] for t in p["link_targets"]),
        "missing_target_references": sum(len(p.get("missing_link_titles", [])) for p in dataset["pages"]),
        "self_loops_removed": sum(any(t["pageid"] == p["pageid"] for t in p["link_targets"]) for p in dataset["pages"]),
        "collection_started_at": min(p["collected_at"] for p in dataset["pages"]),
        "collection_finished_at": max(p["collected_at"] for p in dataset["pages"]),
        "training": training,
    }
    save_json(output / "manifest.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/canonical_articles"))
    args = parser.parse_args()
    client = create_client()
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as http:
        dataset = collect_sample(lambda title: get_resolved_page_links(client, http, title),
                                 args.cache, count=args.count)
    stats = write_outputs(dataset, args.output)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
