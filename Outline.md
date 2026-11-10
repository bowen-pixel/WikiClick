# WikiClick implementation outline

Planning date: November 4, 2026. Target demo: December 1, 2026.

Progress reviewed: **November 9, 2026 (America/Toronto)**. The implementation is ahead of the original milestones, but release readiness and stronger evaluation remain unfinished. The active nested checkout contains the recent collection, training, and plotting commits; 20 tests passed during this review. Treat the schedule below as milestones rather than a claim that every acceptance detail is complete.

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

| Dates (2026) | Original milestone | Status on Nov 9 | Remaining focus |
| --- | --- | --- | --- |
| Nov 5–8 | Environment, API spike, toy solver, module contracts | Core prototype complete; formal result contracts remain provisional | Document actual module boundaries without unnecessary rewrites |
| Nov 9–15 | Cache, bounded collection, canonicalization, graph persistence | Core collection complete ahead of schedule: 100 and 1,100 articles, resumable records, canonical target IDs, validation | Fresh-environment setup check; cover retry/rate-limit behavior; establish CI and a reproducible frozen-data handoff |
| Nov 16–22 | Offline solver, CLI, evaluation | Core behavior implemented; source/target prompts and benchmark reports work | Polish user output, include per-query runtime in the interactive demo, improve errors, and check integration from a clean checkout |
| Nov 23–27 | Optional PyTorch experiment | Initial experiment complete: separate models, held-out destinations, transfer test, BFS/text/neural comparisons and plots | Predefine shared held-out queries for a controlled comparison between model sizes; repeat across seeds and quantify uncertainty. Live-fetch evaluation is optional additional scope |
| Nov 28–30 | Documentation and release preparation | Documentation and measured results exist; release verification pending | Reproduce the demo from a fresh checkout, package/version necessary data and artifacts, finalize limitations and tag a release |
| Dec 1 | Demonstration | Pending | Demonstrate the exact solver and learned-search trade-off using the frozen dataset |

### Verified results so far

- Frozen expanded graph: **1,100 articles and 63,966 unit-weight directed edges**; original 100 articles retained.
- Validated 220,000 larger-graph training rows. Separate 100-article and 1,100-article model checkpoints and training histories exist locally.
- Exhaustive exact analysis covers 1,208,900 distinct ordered pairs: 1,207,801 reachable and 1,099 unreachable. Mean finite shortest distance: **2.4044739158 clicks**. This is reference graph analysis, not an exhaustive neural-search benchmark.
- On 100 held-out larger-graph queries with a 200-expansion budget, BFS reached 97 goals, returning 97 optimal routes; neural guidance reached all 100, returning 69 optimal routes. Mean expansions were 37.23 versus 8.25, respectively; neural inference was slower in memory. These are initial single-seed offline results.
- The 100-article and 1,100-article model comparisons currently use different query sets; do not claim a controlled training-size effect. No live API-fetch speedup has been measured.

### Priorities for November 9–15

1. Use the nested checkout with the recent commits as the working repository; avoid accidentally mixing its virtual environment or generated artifacts with the older outer checkout. Preserve both until deliberate cleanup is requested.
2. Verify installation and a small offline demo from a fresh environment. Decide how another person obtains the exact frozen graph: a versioned artifact or a documented reconstruction plus graph ID, rather than relying on ignored local files.
3. Add automated CI and targeted API retry/rate-limit tests; the current local suite has 20 passing tests.
4. Bring the documented module interfaces and CLI behavior into agreement. Full clean-architecture refactoring is not required to meet the project goal.
5. Specify the next evaluation protocol before further model tuning. Keep a genuinely untouched confirmation set or disclose repeated use of the current test queries.

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

Task status as of November 9:

- [ ] Define page identity, `PageLinks`, and `SearchResult` contracts.
- [x] Set up pinned Python dependencies and local automated tests; hosted CI remains pending.
- [x] Implement continuation and canonical target IDs in the canonical dataset collector; the original small-domain title-only mode retains its documented limitations.
- [x] Implement resumable bounded graph collection with metadata.
- [ ] Implement graph save/load and deterministic test fixture.
- [x] Implement exact shortest-path wrapper and meaningful tests.
- [x] Integrate interactive source/target CLI and benchmark saved-graph queries; user-facing runtime display remains pending.
- [x] Document setup, data scope, and measured results; clean-checkout reproducibility remains pending.
- [x] Train and evaluate the initial navigation models; broader evaluation remains pending.

For each issue, create a short branch such as `codex/wiki-client`, keep changes focused, and open a pull request explaining behavior and validation. Review shared-interface changes before merging. Merge into `main` regularly and update dependent branches. Record AI assistance in `ai_usage.md` as work proceeds.

## 7. Sources

- [NetworkX shortest_path](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html)
- [MediaWiki links](https://www.mediawiki.org/wiki/API:Links)
- [MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)
- [PyTorch quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
