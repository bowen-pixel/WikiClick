# AI usage record

This file records AI assistance used in WikiClick. It distinguishes planning and drafting from implemented and verified functionality. Update it when AI contributes to later code, tests, experiments, or documentation.

## Initial planning and repository documentation

- **Date:** November 4, 2026 (America/Toronto).
- **Tool:** OpenAI Codex, with an earlier planning discussion in the same chat.
- **User request:** Plan a Wikipedia fewest-click project using NetworkX, separate the work into modules and GitHub branches, target December 1, and prepare `README.md`, `Outlin.md`, and `ai_usage.md`.
- **AI contribution:** Explained the directed, unweighted graph model; recommended exact shortest-path search as the baseline; proposed an optional neural-network experiment; drafted the repository overview, module boundaries, milestone schedule, validation goals, and this disclosure. Prepared a Python-oriented `.gitignore` for local/generated files.
- **Sources consulted:** Official NetworkX `shortest_path` and shortest-path algorithm documentation; MediaWiki links, API etiquette, and rate-limit documentation; PyTorch quickstart.
- **Verification at this stage:** Documentation and local-link consistency checks only. No application has been implemented, no Wikipedia graph has been collected, and no model has been trained or evaluated.
- **Human review:** Pending. Repository creation or publication does not imply that the owner has reviewed every technical detail.

The initial illustrative NetworkX snippet is a proposed usage example, not evidence of an implemented application. Performance targets and dataset sizes are planning estimates, not measured results.

## API and shortest-path prototype

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Task:** Use the Wikipedia-API package on a feature branch, extract article links, assign weight 1 to directed edges, and test NetworkX on a small domain.
- **Files:** `api_endpoint.py`, `graph_demo.py`, dependency files, `tests/test_api_and_graph.py`, and documentation updates.
- **Contribution:** Drafted the Python implementation, six offline tests, and setup instructions. Used the package's documented page/link interface; inspected installed package behavior for metadata, pagination, configuration, and exceptions.
- **Validation actually performed:** Installed Wikipedia-API 0.16.0 and NetworkX 3.7 in a local Python 3.13 virtual environment. Six pytest cases passed. Live collection returned five article records; the graph had 13 edges of weight 1. NetworkX found `Graph theory → Shortest path problem → Dijkstra's algorithm` in 2 clicks.
- **Limitations:** Outgoing redirect aliases are matched only when canonical or explicitly requested titles are known. No persistent HTTP cache, full crawler, ML model, or web service was built. Live data is saved locally outside Git. Human review remains pending.

## Ongoing logging template

For each material contribution, add:

```text
Date:
Tool/model (if known):
Task or prompt summary:
Files or pull request affected:
AI contribution:
Human changes and review:
Validation actually performed:
Limitations or unresolved questions:
```

Do not invent model versions, human approvals, test results, or claims that generated work was independently written. If this project is submitted for a course, check the course's actual AI-use rules and adapt this record to its required format.

Do not include passwords, tokens, private chat transcripts, or other sensitive information in this public log. A concise task summary is sufficient unless a specific submission policy requires more.

## Interactive article input

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Contribution:** Added source/target prompts to the graph demo and an optional interactive collection mode with intermediate articles. Kept explicit command-line arguments available. Updated both matching local checkouts and documented usage.
- **Validation:** Prompt tests and offline CLI smoke checks; no additional live Wikipedia collection was needed.
- **Limitations:** Interactive input still searches only the saved graph; entering endpoints does not discover routes outside its selected articles. Human review pending.

## 100-article pilot dataset

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Task:** Collect an initial 100 articles and save clean graph data and candidate-next-article training labels without requiring SQL.
- **Contribution:** Added a resumable seeded sampler, full canonical outgoing target ID extraction through the public MediaWiki API, a JSON graph snapshot, JSONL training export, destination-disjoint splits, metadata, and four additional tests. Source metadata still uses Wikipedia-API; its title-only link interface is supplemented with a public generator query to resolve target redirects efficiently.
- **Validation actually performed:** Collected 100 fully fetched sources with 858 unit-weight directed edges and no failed source fetches. Twelve tests passed. Checked all 84,942 exported labels against NetworkX all-pairs shortest distances, verified graph content hash, valid outgoing candidates, no self-loops, and disjoint destination partitions. Saved a local validation report tied to the graph ID.
- **Limitations:** A topic-biased sample, live responses collected at multiple times, and graph-relative labels; null distances do not imply Wikipedia-wide unreachability. Shared graph and correlated rows mean this is not an unseen-graph benchmark. No model trained, no independent human content audit, and human review pending. Dataset and caches remain local and ignored by Git.

## PyTorch training and 1,000 additional articles

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Task:** Set up PyTorch training with 1,000 additional articles alongside the existing 100-article dataset.
- **Contribution:** Added pinned ML dependencies, stable title hashing, a small distance-regression network, deduplicated pair loading, validation-based checkpoint selection, and offline BFS/text/neural search comparisons. Expanded the resumable sample to 1,100 total sources and capped larger JSONL exports at 200 rows per destination. Added reusable frozen-dataset validation and meaningful model/search/split tests.
- **Validation actually performed:** Seventeen tests passed in both local checkouts. All 220,000 larger-graph labels match forward NetworkX BFS; graph has 1,100 nodes and 63,966 unit-weight edges; all original 100 IDs retained. Trained separate 25-epoch CPU runs on 100 and 1,100 articles, selecting epochs 13 and 8 respectively using validation MAE. Test MAEs were 0.593 and 0.414 clicks. Ran 100-query within-graph comparisons for both models and a frozen-model transfer comparison on new expanded-graph destinations. Detailed measured results are in TRAINING_RESULTS.md and local runs reports.
- **Limitations:** Title-only hashed features; null labels skipped; no optimality guarantee for learned search; biased and overlapping graph samples; offline timings only; no live-fetch speedup claim. Test comparisons across model sizes use different query sets and do not establish a controlled training-size effect. No cloud deployment or model service was created. Human review pending.

## Training and search visualization

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Contribution:** Added a Matplotlib plotting script and dependency file. Generated PNG/PDF learning curves, the fixed Smooth L1 loss shape, and paired offline search-comparison charts with a CSV summary. Documented operational definitions of useful search guidance and limitations.
- **Validation:** Generated charts directly from saved metrics and comparison JSON without retraining. Visually inspected plots and corrected annotation/legend placement. Budget curves derive from recorded deterministic stopping counts within the measured 200-expansion budget.
- **Limitations:** Training Smooth L1 loss and validation MAE are different quantities; separate plots do not measure a numerical train/validation loss gap. No new training runs, uncertainty analysis, or live-fetch benchmark were performed. Human review pending.

## Exhaustive optimal-path analysis

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Contribution:** Added analyze_all_pairs.py and three directed/unreachable/unit-weight tests. Computed exact distances with one BFS per source and saved every distinct ordered pair as compressed CSV plus summary and per-source statistics.
- **Validation:** Three new tests passed. Verified all 1,208,900 exported pair identities in order, no self-pairs, complete row coverage, unreachable count, distance histogram, and sum against summary totals. The mean among 1,207,801 reachable pairs is 2.4044739158 clicks; 1,099 pairs have no finite route in the saved graph.
- **Scope:** Exact reference analysis of the frozen 1,100-node graph, not neural search and not all Wikipedia. Unreachable pairs are excluded from the finite mean. Human review pending.
