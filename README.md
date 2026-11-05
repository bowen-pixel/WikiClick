# WikiClick

Find a route between two English Wikipedia articles using the fewest hyperlink clicks within a collected graph.

**Status:** planning and documentation. The Wikipedia client, graph collector, solver, and demo have not been implemented yet.

**Target:** a reproducible working demo by December 1, 2026. A small neural-network experiment is a stretch goal.

## How it will work

Each article is a node in a simple, directed, unweighted graph. An edge `A → B` means article A links to article B; it does not imply a link back. Each edge costs one click. Repeated links count as one edge, automatic redirects add no click, and search-bar jumps and external websites are excluded.

The first version will collect article-to-article links through the MediaWiki Action API, cache the results, and use NetworkX to find a shortest path:

```python
path = nx.shortest_path(graph, source=start, target=destination, weight=None)
clicks = len(path) - 1
```

This is the intended solver interface, not a runnable application yet. No neural-network training is needed for exact shortest-path search.

**Scope of the answer:** shortest paths are guaranteed only within the saved graph. A partial Wikipedia dataset can miss shorter routes. “No path in this dataset” does not establish that no path exists across Wikipedia.

## Planned components

| Component | Responsibility |
| --- | --- |
| Wikipedia client | Fetch outgoing links; handle pagination, title normalization, redirects, missing articles, and retries |
| Graph and storage | Cache responses; construct, save, and load a directed graph |
| Search | Return a shortest path, click count, status, and runtime |
| CLI demo | Accept two articles and display the route with article URLs |
| Optional ML experiment | Train a small next-link scoring model and compare its navigation with exact search |

Start with a modest saved dataset and an offline CLI. Whole-Wikipedia coverage, a web interface, and neural-network-guided navigation are outside the core December 1 deliverable.

## Project documents

- [Implementation outline and timeline](Outlin.md)
- [AI usage record](ai_usage.md)

## Development workflow

Use one repository and short feature branches such as `codex/wiki-client`, `codex/graph-storage`, and `codex/shortest-path`. Create an issue for each task, agree on shared interfaces, open a pull request, and merge small working changes regularly. Keep `main` usable. A GitHub Project board can be added later to track Todo, In progress, and Done.

Python, NetworkX, an HTTP client, and pytest are the planned baseline. Add PyTorch only when the baseline solver and training dataset are ready. Installation and run instructions will be added when executable code exists.

## Validation goals

- Correct directionality, minimum click counts, zero-click self paths, and missing/unreachable-node handling on small known graphs.
- Mocked API tests for continuation, redirects, missing articles, and transient errors.
- Repeatable searches against a saved graph without requiring live API calls.
- Documented dataset collection time, filtering rules, node/edge counts, and collection limits.

## References

- [NetworkX shortest_path](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html)
- [NetworkX shortest-path algorithms](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html)
- [MediaWiki links API](https://www.mediawiki.org/wiki/API:Links)
- [MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)

Wikipedia requests should use an identifying User-Agent, caching, bounded collection, and appropriate retry/backoff behavior. Any redistributed Wikipedia content must retain the applicable attribution and license information.
