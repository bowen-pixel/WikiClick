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

## Timeline progress review

- **Date:** November 9, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Contribution:** Reviewed the active nested checkout, Git history, saved metrics, and original milestones; updated Outline.md with completed work, explicit remaining acceptance details, and November 9–15 priorities.
- **Validation:** All 20 local tests passed. Confirmed the saved exhaustive-distance summary and initial model/search reports. Did not rerun collection, training, or exhaustive analysis.
- **Limitations:** Local commits and files establish local progress, not remote GitHub synchronization or release readiness. CI, clean-checkout reproducibility, stronger confirmation evaluation, and live-fetch benchmarking are not claimed complete.

## Fresh million-additional-article job (in progress)

- **Date:** November 9, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **User choice:** Fresh Wikipedia data only; no historical SNAP dataset.
- **Contribution:** Inspected the official November 2026 dump manifest and table schemas. Added resumable checksum-verified downloading, streaming SQL-to-CSR import with canonical redirect handling, preservation of original IDs plus a million extra articles, a sampled sparse-graph transfer benchmark, measured-diagram generation, and a local job/status runner.
- **Validation performed:** Three new fixture tests and the full 23-test suite passed. Actual fresh dump download and the dependent local workflow were launched; at the time of this entry the job was still downloading.
- **Not yet claimed:** A completed million-page graph, measured BFS/neural performance, new diagrams, retraining on a million pages, or exhaustive all-pairs evaluation. Final state is recorded in runs/fresh_scale_status.json.

## Fresh dump parser correction

- **Date:** November 9, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Problem:** The numeric parser removed SQL statement semicolons without inserting a delimiter. Multiple INSERT statements in one real dump chunk could concatenate adjacent IDs, triggering a misleading schema error.
- **Change:** Treat statement-ending semicolons as separators, retain strict three-column/count validation, improve the malformed-row error, and add regression tests for multiple statements and invalid column counts.
- **Validation:** All 25 tests passed. Parsed 744,680 real dump records across the first three chunks. An independent regex row-by-row parser matched all 247,323 rows in the previously failing chunk.
- **Execution:** Resumed the local pipeline using existing verified downloads and saved page/target stages. At the progress check it had passed 30 million real hyperlink rows. Import, benchmark, and final diagrams remain pending until the job reports completion.

## Rescaled diagrams and frozen-model interpretation

- **Date:** November 10, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Contribution:** Increased chart resolution and typography, exported individual search panels, changed latency to a logarithmic axis, and autoscaled success axes without altering measurements. Updated the fresh-million guide with completed graph counts and interpretation.
- **Verification:** Regenerated both pilot and fresh-million figures and visually inspected the new combined chart. Read the reports: the frozen neural model reached 7/100 goals versus title similarity 15/100 and BFS 3/100; all failures exhausted the 200-expansion budget. Mean reference query distances differed (2.47 versus 5.67), and caching protocols differed.
- **Scope:** Plotting and explanation only; no new model training, weight updates, or search experiments were performed. Internal neural parameters are distinct from the graph's fixed unit edge costs.

## Second million expansion and supervised weight training (running)

- **Date:** November 10, 2026 (America/Toronto).
- **User intent:** Add another million articles and train on the resulting data, rather than only test frozen weights.
- **Contribution:** Expanded retention to accept a sparse source graph, started a 2,001,100-node import preserving the previous 1,001,100 page IDs, and added exact sampled-label generation, destination-disjoint splits, equivalent faster fixed title hashing, actual Adam fine-tuning, optimizer/parameter-change reporting, shared-query frozen-versus-trained evaluation, and dedicated large-model training plots. Reused the already-downloaded fresh snapshot; no new network download required.
- **Validation:** All 30 tests passed, including a small end-to-end test of exact labels, training parameter changes, and identical held-out queries across both checkpoints. Custom training-history plot generation was exercised successfully.
- **Execution status:** The real expansion and dependent supervised-training workflow were launched. At this entry the real job was still importing; large-graph model updates and performance are not yet claimed. Status: runs/large_training_status.json.
- **Scope:** The plan samples up to 512,000 exact distance pairs from the expanded graph. It does not train on every possible pair or article full text. The existing 1,100-article model initializes a separately saved fine-tuned checkpoint and remains unchanged as control. Human review pending.

## Supervised expanded-graph training completed

- **Date:** November 10, 2026 (America/Toronto).
- **Execution:** Completed a graph of 2,001,100 articles and 56,350,501 edges, then generated sampled exact labels on newly added destination articles. Fine-tuning completed 25 epochs and 39,625 optimizer updates; selected-weight L2 change was 7.25848, with epoch 1 selected by validation MAE.
- **Actual data and metrics:** 405,522 training pairs, 51,200 validation pairs, and 49,158 test pairs. Initial validation MAE 2.76868, selected validation MAE 1.06644, test MAE 0.90289; constant-distance test MAE 0.57463. Thus training does not establish superiority to the constant baseline or better routing.
- **Artifacts:** Saved separate best.pt, last.pt, progress and metrics; generated actual large-graph learning curves. The original checkpoint remains untouched. Paired routing evaluation was still running at this entry; no final routing result is claimed yet.

## Experiment audit and readable failure diagrams

- **Date:** November 10, 2026 (America/Toronto).
- **Contribution:** Audited saved experiments, distinguished graph size from labelled training-pair count, and added `plot_training_audit.py` plus `TRAINING_AUDIT.md`. No training exists at exactly 10 or 1,000 articles; the 1,001,100 graph used frozen transfer. Re-scored both checkpoints on the identical 49,158 held-out distance pairs without updating weights or choosing a new checkpoint.
- **Measured result:** Frozen MAE 2.502322, fine-tuned MAE 0.902893, constant train-mean MAE 0.574634 clicks. The matched two-million routing reports are complete: BFS, title similarity, frozen neural and fine-tuned neural all reached 0/100 goals at 200 expansions; every failure exhausted that budget. These results do not establish better routing or superiority to the constant distance baseline.
- **Plots:** Replaced overlapping all-zero success curves with count tables and failure-reason bars. Added matched error, matched routing outcome and fine-tuning learning-curve PNG/PDF outputs. Visually checked the generated charts. Cross-size runs use different test queries and are not a controlled scaling study. No test-time training has been implemented.

## 1,000-article next-link ranking and correction pilot

- **Date:** November 11, 2026 (America/Toronto).
- **User request:** Run a 1,000-article supervised next-link ranking model with a correction round and compare it against the raw neural model.
- **Implementation:** Added `next_link_experiment.py`, `rank_navigate.py`, `plot_next_link.py`, algorithm tests, and `NEXT_LINK_1000.md`. Used a separate deterministic 1,000-node, 54,404-edge induced subset of the earlier 1,100-article graph. Downloaded current public introductions and visible category memberships; cached source/revision information. Found 999 introductions and 833 retained category hyperedges covering 860 articles.
- **Model scope:** Retained the original title-only distance architecture as a freshly trained baseline. Added grouped optimal-next-link training and a residual context head using fixed intro hashes and category-hypergraph pooling. No pretrained language encoder, free-form relationship extraction, learned hypergraph encoder, RL, Coconut or inference-time weight updates are claimed.
- **Correction and evaluation:** Used 800/100/100 disjoint destination partitions, 8,000/1,000/1,000 decisions, and three seeds. Mined training mistakes only, resampled them at weight 4, and compared with equal-update ordinary continuation. Checkpoints selected by validation loss before test labels were evaluated. All three correction and continuation stages selected epoch 0: proposed updates were rejected, and the previous semantic ranker was retained.
- **Measured results:** Mean route success across seeds: raw 91.53%, title ranker 98.50%, text/category ranker 98.57%. Optimal-route rates: 51.17%, 66.17%, 68.13%. Correction provided zero additional gain. The semantic-versus-title route-success difference was only +0.067 percentage points, with conditional destination-clustered 95% interval [-0.933, +1.067]; no added semantic success gain is established. Raw versus the full pipeline differs in training duration, available features, and parameter count and is not an isolated causal test of the objective.
- **Validation:** All 34 tests passed. Visually inspected charts and exercised local inference: the corrected (rolled-back) policy reached ENIAC from Distributed artificial intelligence in 3 clicks; the raw policy exhausted its 20-click limit on that illustrative test query. Aggregates include all 1,000 matched queries, not just that example. Local training and evaluation completed in about 191 seconds, excluding collection and plotting. Existing large-graph artifacts were preserved.
