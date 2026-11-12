# Million-article semantic next-link ranking and correction

Started November 11, 2026 (America/Toronto). **Stopped after a network/DNS failure; not yet measured.**
The workflow downloads semantic source data, trains actual weights, performs
validation-controlled correction, and then generates measured comparison diagrams.
No pilot results are presented as million-node results.

## Fixed graph and fresh semantic sources

- Reuse `data/graphs/fresh_million`: **1,001,100 nodes and 14,742,079 directed unit-cost links**.
  This is the earlier million additional articles plus the retained 1,100, not a new graph import.
- Download the matching **20261101** English Wikipedia article-text dump: **71 shards,
  25,899,473,885 compressed bytes**. The previous downloads supplied hyperlink tables,
  not article prose or category memberships. Each new text shard is checked against
  Wikimedia's published size/SHA1 before use. Partial downloads resume with validated ranges.
- Disk-saving mode retains the gzip-compressed records for selected graph articles,
  verifies their integrity and provenance, then discards that shard's bulk `.bz2` copy.
  Resuming reuses the verified extraction instead of downloading that shard again.
  The largest raw shard is about **614 MB**. This reduces retained raw-dump space,
  but the full run still transfers 25.9 GB and needs additional space for extracted
  records, label caches and numerical features. The original hyperlink graph is untouched.
- Stream XML rather than unpacking the whole corpus. Retain source records only for
  page IDs already in the fixed graph, with revision IDs, timestamps and dump filenames.
- Extract a bounded lead passage with approximate markup cleanup, plus explicit
  wikitext category memberships. This omits template-inherited categories and may
  include hidden or maintenance categories; it is not identical to the small pilot's
  API `!hidden` category view. Sampled real records were previewed for parser validation;
  unverified partial-file previews are never used in training.
- Build fixed 128-dimensional title features, 64-dimensional word-hashed text features,
  and 64-dimensional category-group context. Hyperedges have 2–1,000 members; storage
  stays sparse, with no clique expansion or dense million-by-million matrix.
  At least 95% article-source coverage is required before semantic training proceeds.

These are lightweight text/category features, not pretrained language embeddings,
free-form relation extraction, learned hypergraph message passing or Coconut.
The text feature version differs from the 1,000 pilot's character hashing and is
recorded explicitly. All article features are unlabeled transductive inputs; true
test distances and route outcomes are not policy inputs.

## Supervision and training

Exact labels are already generated on the full directed graph:

| Split | Distinct destinations | Ranking decisions |
|---|---:|---:|
| Training | 64 | 8,192 |
| Validation | 16 | 1,024 |
| Test | 16 | 2,048 |

Each decision lists all outgoing candidates and all neighbors that reduce exact
remaining distance by one. Distances are computed by compiled reverse-graph
shortest-path traversal; unreachable neighbors remain negative choices. Destinations
do not overlap between splits and are outside the old 1,100 articles. This is sampled
supervision on a million-node graph, not a million independent training examples or
all-pairs supervision.

For each of seeds **207, 208 and 209**:

1. Train a fresh raw title-only distance model for 10 epochs, using up to 60,000
   decision-derived reachable pairs and validation MAE checkpoint selection.
2. Initialize separate title-only and text/category residual rankers from that raw
   checkpoint. Train each for 10 additional epochs with grouped next-link cross-entropy.
3. Identify mistakes in the **training decisions only**. Sample those decisions with
   weight 4 during a 5-epoch correction round.
4. Run an ordinary 5-epoch continuation from the identical semantic checkpoint with
   the same number of optimizer updates as the correction control.
5. Select by validation loss, including epoch 0 (retain the starting checkpoint) if
   every proposed update is worse. A correction round is not assumed to improve anything.

Two CPU threads are used. Features are memory-mapped. No million-node graph tensor is
passed through the MLP for each optimizer update. Complete seeds are reusable; an
interrupted seed restarts training rather than resuming the optimizer mid-epoch.

## Matched evaluation after training

Only after every seed/condition is trained and checkpoint-selected, evaluate:

- Optimal-next-link accuracy on all 2,048 held-out decisions.
- 20-click navigation on 96 identical reachable test queries.
- BFS and each learned frontier-search policy on those same 96 queries, with expansion
  limits **200, 2,000 and 10,000**, each also subject to **3 seconds per query**.
- Goals reached, minimum-click routes, expansions, candidates scored and latency.

One search prefix supplies the three expansion-budget observations. Exact test-query
lengths are independently checked before benchmarking, and every successful route is
checked against actual directed hyperlinks. No semantic group creates a clickable edge.
Learned scores prioritize exploration; only BFS guarantees the minimum click count.
All title/text/category vectors are precomputed and indexing costs are separate from
search latency. Identical retained checkpoints explicitly reuse measured outcomes and
timings rather than presenting duplicate measurements as independent runs.

Raw-versus-refined comparisons differ in training duration and available features;
the semantic residual head also adds parameters. The equal-update continuation is
the relevant correction control. This is not an isolated causal test of the ranking
loss, or a controlled learning curve across different graph sizes.

## Check progress

From the nested repository:

```bash
cd /Users/bowenguang/Documents/ChatGPT/WikiClick/WikiClick
.venv/bin/python -m codes.workflows.million_ranking_status
tail -f models/next_link/ranking_million/workflow.log
```

`workflow.json` records the overall worker and final `complete`/`failed` state.
`status.json` records the current detailed stage. The status helper totals verified
extracted source bytes separately from the raw archives/partials currently on disk,
without making network requests. Downloading requires an
internet connection; later processing and training are local. The computer must stay
running for the worker to make progress.

Do not start a duplicate worker. After a stopped or failed workflow, restart with:

```bash
.venv/bin/python -m codes.workflows.run_million_ranking
```

To pause a running workflow from a separate Terminal tab:

```bash
.venv/bin/python -m codes.workflows.run_million_ranking --pause
```

This verifies the recorded controller and its stage worker before stopping them,
marks progress `paused`, and preserves cached data and partial downloads. The normal
command without `--pause` resumes the workflow later. If the job runs in the foreground,
Control+C in that job's terminal also stops it; Control+C in a `tail -f` monitoring
terminal only stops the monitor.

The workflow lock prevents concurrent controller instances. Verified shards, completed
extractions, exact labels and completed training seeds are reused. Existing original
graphs and earlier checkpoints are not overwritten.

The current worker has been stopped, and both status files are marked `paused`.
The first completed bulk archive was removed only after its selected-article gzip
extraction and original checksum were verified. The next partial download remains
available to resume. No download or training was restarted during compaction.

### Network/DNS failures on resume

`httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known`
means the request could not resolve the download host. It is not a model, RAM or
training-loss error. Reconnect to a working internet connection and rerun the normal
resume command. A saved partial download is not deleted by a connection failure.

Resume now reuses the pinned local manifest instead of requesting it again. Repeated
connection failures produce a concise message and `waiting_for_network` status,
with exit code 75, rather than nested downloader/controller tracebacks. This handles
the failure safely; it does not repair internet or DNS connectivity. The workflow
stops rather than waiting indefinitely or automatically resuming while disconnected.

## Outputs when complete

- `models/next_link/ranking_million/results.json`: measured per-seed comparison.
- `models/next_link/ranking_million/aggregate.json`: seed-averaged metrics.
- `models/next_link/ranking_million/selected_checkpoints.json`: whether correction was retained.
- `figures/next_link/ranking_million/`: PNG/PDF branching, optimal-route and next-link plots.
- `MILLION_RANKING_RESULTS.md`: generated results and limitations.

These final results/diagrams do not exist merely because data collection has started.
The pipeline generates them automatically after actual model training and evaluation.
The new implementation passed **41 tests**, including an isolated synthetic end-to-end
test of sparse label generation, feature pooling, training, correction and shared-route
evaluation. That test verifies implementation, not million-node effectiveness.
