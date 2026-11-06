# WikiClick

Find a route between two English Wikipedia articles using the fewest hyperlink clicks within a collected graph.

**Status:** working API and graph prototype, plus a cached 100-article pilot and graph-relative training labels. The small-domain prototype remains available. Larger-scale collection and neural-network training remain future work.

**Target:** a reproducible working demo by December 1, 2026. A small neural-network experiment is a stretch goal.

## How it will work

Each article is a node in a simple, directed, unweighted graph. An edge `A → B` means article A links to article B; it does not imply a link back. Each edge costs one click. Repeated links count as one edge, automatic redirects add no click, and search-bar jumps and external websites are excluded.

The first version will collect article-to-article links through the MediaWiki Action API, cache the results, and use NetworkX to find a shortest path:

```python
path = nx.shortest_path(graph, source=start, target=destination, weight=None)
clicks = len(path) - 1
```

The prototype explicitly stores `weight=1` on every edge and calls `nx.shortest_path(..., weight="weight")`. This minimizes clicks just as the unweighted call above does. No neural-network training is needed for exact shortest-path search.

## Run the prototype

Use Python 3.13 (the version used for the initial verification).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python api_endpoint.py
python graph_demo.py
python -m pytest -q
```

Running `python graph_demo.py` now lists the saved graph's articles and prompts for `Enter source:` and `Enter target:`. Invalid titles are rejected and prompted again. Pass both `--source` and `--target` to skip prompts, or pass one to prompt only for the other.

To choose a new graph interactively, run `python api_endpoint.py --interactive`. Enter the source, target, and optional intermediate articles (one per line, then a blank line to finish). Next run `python graph_demo.py` and choose the endpoints from the saved graph. Adding only source and target produces at most a direct route; longer routes need intermediate articles in the dataset. Collection does not automatically discover those intermediates.

`api_endpoint.py` is the Wikipedia **client access module**, not an HTTP server. It uses the package's page objects, metadata, and links methods against `https://en.wikipedia.org/w/api.php`. It fetches only the five selected articles, follows outgoing-link pagination through the library, and saves plain JSON to `data/graphs/small_domain.json` (ignored by Git). It does not recursively crawl outgoing links or fetch article text.

`graph_demo.py` reads that saved JSON without network access, uses page IDs as nodes, and retains only links among the selected articles. Every edge has weight 1. To choose another small domain:

```bash
python api_endpoint.py "Graph theory" "Shortest path problem" "Dijkstra's algorithm" --output data/graphs/custom.json
python graph_demo.py --data data/graphs/custom.json --source "Graph theory" --target "Dijkstra's algorithm"
```

The collector accepts 1–20 unique article titles. This is a source-page limit, not an HTTP-request limit: pagination and retries may require additional requests. Sources are fetched sequentially with an identifying User-Agent, timeout, limited retries, and `maxlag`. Missing articles fail collection instead of silently producing empty records. Rerunning collection makes fresh requests; running the graph demo reuses the saved data.

**Live result on November 5, 2026:** five articles, 13 directed edges, and the route `Graph theory → Shortest path problem → Dijkstra's algorithm` in 2 clicks. Wikipedia changes, so future collections can differ.

**Prototype limitation:** source redirects are resolved, but outgoing links are matched only against selected canonical titles and explicitly requested aliases. Other redirect aliases can be omitted. This is the shortest path in the constructed graph, not a guarantee that every possible link or redirect between those articles has been captured. Namespace-0 API links may include template-generated links rather than only article body links. Full canonicalization and collection metadata will be expanded in later work.

**Scope of the answer:** shortest paths are guaranteed only within the saved graph. A partial Wikipedia dataset can miss shorter routes. “No path in this dataset” does not establish that no path exists across Wikipedia.

## Planned components

| Component | Responsibility |
| --- | --- |
| Wikipedia client | Fetch outgoing links; handle pagination, title normalization, redirects, missing articles, and retries |
| Graph and storage | Cache responses; construct, save, and load a directed graph |
| Search | Return a shortest path, click count, status, and runtime |
| CLI demo | Accept two articles and display the route with article URLs |
| Optional ML experiment | Train a small next-link scoring model and compare its navigation with exact search |

## 100-article dataset and training rows

Run `python sample_dataset.py` to collect a linked sample of 100 articles, resume from completed article records in `data/cache/canonical_articles/`, and export files under `data/graphs/sample_100/`. It starts from ten listed topic seeds and follows outgoing links in breadth-first order with deterministically shuffled neighbors. This sample is biased toward those seeds; it is not uniform sampling of Wikipedia. Reusing the cache reproduces the collected sample; live Wikipedia can change across fresh collections.

For this collector, Wikipedia-API supplies source metadata. A public MediaWiki `generator=links` query resolves outgoing redirects to existing namespace-0 page IDs, with continuation, retries, and deduplication. The graph contains only the 100 fully fetched sources and edges between their IDs; external targets remain boundary information in `articles.json`. A source whose collection fails is never silently treated as a page without links. Completed records are cached, while network errors stop the run so it can be resumed.

| File | What it contains |
| --- | --- |
| `articles.json` | Canonical source articles, URLs, per-source timestamps, complete canonical outgoing targets, missing-link titles, and sampling settings |
| `graph.json` | Frozen directed nodes and edges, each with weight 1; a SHA-256 graph identifier |
| `train.jsonl`, `validation.jsonl`, `test.jsonl` | One labeled candidate-next-click example per line, split by destination article |
| `manifest.json` | Graph counts, collection dates, connectivity, split destination IDs, row counts, and an example |

**JSON is sufficient at this scale; no SQL setup is needed.** Keep graph data separate from training examples so labels can be regenerated. JSONL can be processed one line at a time without loading all training rows into memory. Generated graph/cache files are ignored by Git and remain local.

Each row has `current_article`, `destination`, `candidate_next_article` (each containing a page ID and title), `label_distance`, `reachable_in_graph`, `split`, and `graph_id`. Every candidate is a real outgoing neighbor of the current article in the graph. The numeric label is the shortest distance **from candidate to destination**, excluding the click from current to candidate. A candidate equal to the destination has label 0; selecting it takes one click from the current article. Terminal current states are omitted. `null` means no route within this saved graph, not unreachable across Wikipedia; handle these separately from numeric regression labels.

Destinations are partitioned 80/10/10 with a fixed random seed before rows are written. All rows for one destination stay in its partition. This is transductive evaluation on a shared fixed graph, not a test on unseen articles or unseen graphs. Do not use shortest-distance labels or held-out destination labels as model input features. No model has been trained yet.

To search this sample interactively:

```bash
python graph_demo.py --data data/graphs/sample_100/articles.json
```

The graph reflects live API responses collected at different times, not one atomic Wikipedia revision. Its exact distances apply only to the saved graph. This is a validated pilot dataset, not a claim of perfect Wikipedia-wide coverage or an independent content audit.

Initial collection on November 5, 2026 (America/Toronto): **100 articles, 858 edges, one weak component, and 8 strongly connected components** (largest: 93 articles). There were no failed source collections. Excluded boundary links and missing targets are counted in the manifest. All 84,942 exported row labels were checked against NetworkX all-pairs distances, and destination partitions were checked for overlap. There are 68,093 training rows, 8,351 validation rows, and 8,498 test rows; 2,056 rows have a null distance. Rows share one graph and are highly correlated; row count is not a measure of independent sample size. The local `validation_report.json` records this structural/label verification for the graph ID.

Start with a modest saved dataset and an offline CLI. Whole-Wikipedia coverage, a web interface, and neural-network-guided navigation are outside the core December 1 deliverable.

## Project documents

- [Implementation outline and timeline](Outline.md)
- [AI usage record](ai_usage.md)

## Development workflow

Use one repository and short feature branches such as `codex/wiki-client`, `codex/graph-storage`, and `codex/shortest-path`. Create an issue for each task, agree on shared interfaces, open a pull request, and merge small working changes regularly. Keep `main` usable. A GitHub Project board can be added later to track Todo, In progress, and Done.

The prototype uses Python, Wikipedia-API, NetworkX, and pytest. Add PyTorch only when the baseline solver and training dataset are ready.

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
- [Wikipedia-API package and usage examples](https://pypi.org/project/Wikipedia-API/)

Wikipedia requests should use an identifying User-Agent, caching, bounded collection, and appropriate retry/backoff behavior. Any redistributed Wikipedia content must retain the applicable attribution and license information.
