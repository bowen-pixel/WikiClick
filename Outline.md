# WikiClick implementation outline

Planning date: November 4, 2026. Target demo: December 1, 2026.

This schedule assumes approximately 8–12 hours per week. The tables describe the target architecture. As of November 5, the small-domain prototype, resumable canonical-ID sampler, 100-article pilot, and title-feature PyTorch training/search comparison pipeline are implemented. Expansion to 1,100 articles preserves the pilot and creates separate model artifacts. See `TRAINING.md` for reproducible commands and evaluation limitations. Full Wikipedia coverage, richer semantic features, and live-fetch benchmarking remain future work.

## 1. Minimum deliverable

A command-line program accepts a starting article and destination, searches a reproducible saved graph, and displays a shortest route with clickable article URLs, the click count, and runtime. It clearly identifies the dataset used and reports missing nodes or no route in that dataset without making claims about all Wikipedia.

Use a simple directed graph (`networkx.DiGraph`) and unit edge costs. No hypergraph is required. Treat API-provided namespace-0 article links as the initial link definition; do not claim this exactly matches a body-text-only Wikipedia game. Normalize aliases and resolve redirect targets to canonical articles before constructing edges. Exclude missing targets and self-loops, and deduplicate edges.

## 2. Module boundaries

These are proposed interfaces to settle before separate implementation branches begin.

| Module | Proposed interface | Contract |
| --- | --- | --- |
| `wiki_client.py` | `get_links(title) -> PageLinks` | Canonical source identity and complete outgoing canonical article links; failures remain distinguishable from an empty link list |
| `graph_store.py` | `build_graph(...)`, `save_graph(...)`, `load_graph(...)` | Directed graph plus metadata; explicit page/request/time budgets |
| `search.py` | `find_path(graph, source, target) -> SearchResult` | Path, clicks, status, and runtime; no network side effects |
| `cli.py` | command-line entry point | Validate inputs, load dataset, call solver, format results |
| `ml/` (optional) | dataset, train, evaluate scripts | Reproducible experiment separate from exact solver |

Use stable page IDs as internal identities and retain canonical titles for display. Define `PageLinks` and `SearchResult` together before implementation. Suggested statuses: `found`, `source_missing`, `target_missing`, and `no_path_in_dataset`. For failure results, clicks should be absent rather than zero.

Keep dependency direction simple: the CLI calls the search and storage modules; collection uses the Wikipedia client. The search module only receives a graph and normalized node identities, so it can be tested independently.

## 3. Dataset collection

1. Use the English Wikipedia MediaWiki Action API at `https://en.wikipedia.org/w/api.php`.
2. Request `action=query`, `prop=links`, `plnamespace=0`, `pllimit=max`, and JSON output; follow every continuation response within the collection budget.
3. Handle title normalization and redirect resolution separately where needed; merely fetching links does not guarantee canonical targets.
4. Cache responses and identify the client with a meaningful User-Agent. Start with sequential requests; respect `Retry-After` and back off on transient errors. Use `maxlag` for background collection.
5. Start with several seed articles and a bounded collection. An initial target of 500–2,000 articles is provisional; measure API cost before choosing the final budget.
6. Record seeds, collection timestamp, API parameters, graph counts, limits, failed/incomplete pages, and filtering rules. A discovered article whose links have not been fetched is not evidence of a dead end.
7. Freeze a dataset for repeatable evaluation. Keep raw caches and large generated graphs out of ordinary Git commits; commit a small deterministic fixture and metadata instead.

## 4. Timeline and acceptance criteria

| Dates (2026) | Work | Acceptance criteria |
| --- | --- | --- |
| Nov 5–8 | Environment, module contracts, API spike, toy solver | Fetch complete outgoing links for one article; demonstrate a correct shortest path on a hand-built directed graph |
| Nov 9–15 | Cache, bounded collection, canonicalization, graph persistence | Save and reload a documented dataset; API pagination and failure cases covered by mocked tests |
| Nov 16–22 | Integrate offline solver and CLI; evaluate | Route and click count displayed; directionality, self paths, unreachable nodes, and missing inputs handled; benchmark pairs recorded |
| Nov 23–27 | Optional small PyTorch experiment | Reproducible training run and held-out evaluation against exact shortest paths |
| Nov 28–30 | Integration fixes, documentation, release preparation | A fresh checkout can follow documented setup and reproduce the demo; limitations and results written down |
| Dec 1 | Demonstration | Working core solver, dataset metadata, example output, and tagged release |

If time is limited, defer ML first, then interface polish. Preserve the exact solver, reproducible data, and honest reporting.

## 5. Optional neural-network experiment

Research question: can a small model rank outgoing links toward a destination and reduce search work while keeping routes short?

- Generate shortest-distance labels using the exact solver on the frozen graph. For a fixed destination, reverse-graph BFS can label distances efficiently; avoid computing every pair unnecessarily.
- A basic model can score candidate links using text-similarity and graph features. Compare with a simple non-neural heuristic before claiming an ML benefit.
- Split training, validation, and test data before fitting features or tuning. Hold out destination articles for a stronger generalization check. State that all labels are relative to the saved graph.
- Restrict navigation to valid outgoing links. Use a visited set and explicit step/search budgets to handle cycles and failed navigation.
- Report success rate, excess clicks over the exact shortest route, nodes explored, and wall-clock runtime including model inference. Report failures separately.
- Learned scores do not automatically preserve shortest-path guarantees. Do not present an unconstrained learned heuristic as an admissible A* heuristic.

## 6. GitHub tasks and merging

Use the repository for code, documents, issues, and pull requests. A GitHub Project is an optional task board, not a replacement for the repository.

Suggested first issues:

- [ ] Define page identity, `PageLinks`, and `SearchResult` contracts.
- [ ] Set up Python dependencies and basic automated checks.
- [ ] Implement Wikipedia client with continuation and canonicalization.
- [ ] Implement caching and bounded graph collection with metadata.
- [ ] Implement graph save/load and deterministic test fixture.
- [ ] Implement exact shortest-path wrapper and meaningful tests.
- [ ] Integrate CLI and benchmark saved-graph queries.
- [ ] Document reproducible setup, data scope, and demo results.
- [ ] Stretch: train and evaluate a small navigation model.

For each issue, create a short branch such as `codex/wiki-client`, keep changes focused, and open a pull request explaining behavior and validation. Review shared-interface changes before merging. Merge into `main` regularly and update dependent branches. Record AI assistance in `ai_usage.md` as work proceeds.

## 7. Sources

- [NetworkX shortest_path](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html)
- [MediaWiki links](https://www.mediawiki.org/wiki/API:Links)
- [MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)
- [PyTorch quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
