# Fresh million-additional-article benchmark

Started November 9, 2026 (America/Toronto). The user explicitly chose fresh Wikipedia data; the historical 2011 alternative is not used.

## Current run

The initial job is complete: 1,001,100 article nodes and 14,742,079 unit-weight directed edges. The measured comparison is `runs/fresh_million_comparison.json`; diagrams are in `runs/fresh_million_figures/`. The existing 1,100-article checkpoint was frozen: this run did not train or update model weights on the million-page graph. Historical job state and logs remain available in `runs/fresh_scale_status.json` and `runs/fresh_scale_job.log`.

The job uses the dated English Wikipedia **20261101** database dump. The relevant tables were completed during the November dump run (file timestamps around November 7). This is a recent frozen bulk snapshot, not a promise of live November 9 content. Source manifest: https://dumps.wikimedia.org/enwiki/20261101/dumpstatus.json.

Required compressed files total **11,257,940,676 bytes**:

| Table | Compressed bytes | Purpose |
| --- | ---: | --- |
| page | 2,426,942,287 | Article IDs, namespaces, canonical titles, redirect flags |
| redirect | 188,211,625 | Local redirect chains |
| linktarget | 1,452,254,955 | Map hyperlink target IDs to article titles |
| pagelinks | 7,190,531,809 | Directed hyperlink edges |

Downloads are streamed, resumable, and verified against the official manifest SHA-1 and size. SQL is parsed as data; it is not executed, and no SQL database server is required. Full decompressed tables are never saved.

## Graph and model protocol

1. Retain all 1,100 original article page IDs if they remain canonical articles in this snapshot. Fail explicitly if any original ID cannot be retained; do not merge stale old graph edges into the new snapshot.
2. Reservoir-sample **1,000,000 additional nonredirect namespace-0 articles**, using seed 207. Target total: **1,001,100 nodes**. This uniform induced sample differs from the earlier BFS-topic sample and can contain disconnected pages.
3. Resolve local redirect chains and map link targets to selected canonical article IDs. Remove self-loops and duplicate edges; each edge has cost 1.
4. Store compressed sparse row arrays and page/title tables in `data/graphs/fresh_million/`. No enormous JSON graph object or all-pairs training export is constructed.
5. Test the **frozen existing 1,100-article model**; do not retrain or tune it on these evaluation answers. This is a transfer/scale test. Destination page IDs exclude all original model graph IDs.
6. Preselect 100 graph-reachable directed queries at least two clicks apart. Compute exact reference distances with compiled sparse shortest-path routines. This is a sampled-query benchmark, not an exhaustive average over roughly one trillion possible pairs.
7. Compare BFS, title similarity, and neural greedy best-first on identical queries, with a **200-expansion and 10-second budget per method/query**. Title features start cold for each guided method/query, and their computation plus inference is included in measured runtime. There are no API calls during search.
8. Only after measured results exist, generate PNG/PDF comparison diagrams and CSV metrics in `runs/fresh_million_figures/`.

Timing of compiled reference calculation is separate from algorithm search timing. This is a graph traversal benchmark, not live webpage-fetch latency. The neural method has no shortest-path guarantee. Feature cold-start costs differ from the shared-preprocessing pilot benchmark, so compare methods within this run rather than treating its timing as directly equivalent to the earlier reports.

## Reproduce or resume

```bash
source .venv/bin/activate
python -m pip install -r requirements-scale.txt
python run_fresh_scale.py
```

Do not start a second download/import job while the current one is active. The current orchestration was launched with `--wait-for-download` because an existing verified downloader already owned the partial files. The machine must remain running for local processing to continue. If the job fails, inspect the status/error and log before restarting; completed download files and importer stage checkpoints are reusable.

Individual stages:

```bash
python download_fresh_dump.py
python import_fresh_dump.py
python benchmark_fresh_scale.py --queries 100 --budget 200 --seconds-per-query 10
python plot_results.py --search-only --comparison runs/fresh_million_comparison.json --output runs/fresh_million_figures
```

The importer and sparse search passed three new tests; the full local suite passed **23 tests** before this job was launched. These are implementation checks on small fixtures, not proof that the real million-page import has finished or that the neural model performs well at this scale.

## Initial measured result and interpretation

On 100 paired queries (mean exact optimal distance 5.67 clicks), BFS reached 3 goals, title similarity reached 15, and the frozen neural model reached 7. They returned 3, 5, and 3 optimal routes, respectively. All failures were expansion-budget exhaustion; no time-limit failures occurred. Thus this title-only pilot model transfers poorly, and it did not beat the non-neural title heuristic on success rate.

The earlier 1,100-node benchmark had mean optimal query distance 2.47 clicks and shared title-feature preprocessing. The million-node experiment used different queries, uniform-sample graph construction, and cold per-method feature calculations. The drop is evidence of weak transfer under this protocol, not an isolated causal estimate of graph-size effects. Latency includes feature computation, so profiling is needed to separate preprocessing from neural inference.

Rescaled diagrams use larger fonts, individual panels, a logarithmic latency axis, and success-rate axes sized to the observed rates. The success axes have different ranges across graph sizes; compare labeled numeric rates rather than bar heights across separate images. Regeneration reads saved measurements only; it does not retrain or rerun searches.

```bash
python plot_results.py --search-only --comparison runs/fresh_million_comparison.json --output runs/fresh_million_figures
open runs/fresh_million_figures/route_quality.png
open runs/fresh_million_figures/search_latency.png
```

For adaptation, generate sampled exact labels on the large graph with training/validation destinations separated from a new confirmation test set, train or fine-tune on training batches, select a checkpoint using validation, and evaluate before deploying the new version. Failures used for training cease to be valid held-out tests. Internal neural parameters can change; graph edge costs remain 1 to preserve the fewest-click objective. More epochs on the same small dataset alone do not supply large-graph knowledge.
