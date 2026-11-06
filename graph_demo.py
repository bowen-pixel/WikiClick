"""Build a small directed graph from saved API data and find a fewest-click route."""

import argparse
import json
from pathlib import Path

import networkx as nx


def title_key(title):
    """Match spaces/underscores and Wikipedia's first-letter normalization."""
    title = title.replace("_", " ").strip()
    return title[:1].upper() + title[1:]


def build_graph(dataset):
    """Use page IDs as nodes; retain only edges among selected, existing pages.

    Match canonical titles and the aliases explicitly requested in the dataset.
    Other redirect aliases are not resolved by this first bounded experiment.
    """
    graph = nx.DiGraph()
    aliases = {}
    for page in dataset["pages"]:
        node = page["pageid"]
        graph.add_node(node, title=page["title"], url=page["url"])
        aliases[title_key(page["title"])] = node
        aliases[title_key(page["requested_title"])] = node
    for page in dataset["pages"]:
        source = page["pageid"]
        for title in page["links"]:
            target = aliases.get(title_key(title))
            if target is not None and source != target:
                graph.add_edge(source, target, weight=1)
    return graph, aliases


def find_path(graph, aliases, source, target):
    """Return IDs for a minimum-click path; every stored edge has weight 1."""
    for title in (source, target):
        if title_key(title) not in aliases:
            raise ValueError(f"Article is outside this dataset: {title}")
    return nx.shortest_path(
        graph, source=aliases[title_key(source)],
        target=aliases[title_key(target)], weight="weight",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/graphs/small_domain.json"))
    parser.add_argument("--source", default="Graph theory")
    parser.add_argument("--target", default="Dijkstra's algorithm")
    args = parser.parse_args()
    try:
        data = json.loads(args.data.read_text(encoding="utf-8"))
        graph, aliases = build_graph(data)
        print(f"Directed graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print("Each edge has weight 1. Scope: selected articles and matched titles only.")
        print("Edges:")
        for source, target in sorted(graph.edges):
            print(f"  {graph.nodes[source]['title']} -> {graph.nodes[target]['title']} (weight=1)")
        path = find_path(graph, aliases, args.source, args.target)
    except nx.NetworkXNoPath:
        parser.exit(2, "No path in this dataset; a route elsewhere on Wikipedia may exist.\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot run demo: {error}\nRun api_endpoint.py first to collect the dataset.\n")
    print("\nShortest route within this graph:")
    for node in path:
        print(f"  {graph.nodes[node]['title']} — {graph.nodes[node]['url']}")
    print(f"Clicks: {len(path) - 1}")


if __name__ == "__main__":
    main()
