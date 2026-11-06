# Training and comparing WikiClick

## What is being trained

The first model predicts the number of clicks remaining **from candidate article B to destination D**. At current article A, only its real outgoing neighbors are candidates. The current-to-candidate click is not part of the distance label. A candidate equal to the destination has distance zero.

The initial runs are complete. See [TRAINING_RESULTS.md](TRAINING_RESULTS.md) for measured results and the local `runs/` directory for checkpoints, metrics, and detailed query reports.

The inputs are fixed, normalized hashed word/character features of article titles. Page IDs identify records but are not model inputs. No graph shortest distances, destination test labels, or route answers are input features. This lets the same feature function represent titles outside the training graph without fitting a vocabulary on the test data. Title hashing is a deliberately simple baseline; it is not a pretrained semantic language model and cannot be expected to understand article content deeply.

The PyTorch model has layers `512 → 64 → 32 → 1`, ReLU hidden activations, and a Softplus output. Training uses CPU, Adam, batches of 256, Smooth L1 loss, and a fixed random seed. No GPU or server deployment is required.

## Install

From the checkout you actually use, activate its environment and install:

```bash
source .venv/bin/activate
python -m pip install -r requirements-ml.txt
python -m pytest -q
```

## Existing 100 articles

```bash
python train_model.py --data data/graphs/sample_100 --output runs/model_100 --epochs 25
python compare_search.py --data data/graphs/sample_100 --checkpoint runs/model_100/best.pt --output runs/comparison_100.json --queries 100 --budget 200
```

`best.pt` contains weights and feature/graph metadata. `metrics.json` records each epoch, selected epoch, validation/test mean absolute error, a constant training-mean baseline, selected pair counts, and versions. The best epoch is chosen using validation MAE; test labels are loaded only after model selection. Running experiments repeatedly and choosing settings based on test outcomes would invalidate that test's role as an untouched benchmark.

Distance B→D does not depend on A, so repeated `(candidate,destination)` pairs are deduplicated before training. Unreachable (`null`) labels are counted and skipped for this regression experiment, not replaced with zero. Therefore the model is not trained to detect unreachability. By default each split uses at most 30,000 unique reachable pairs; this is a cap, not a promise of that many unique examples.

## Add 1,000 more articles (1,100 total)

Preserve the original `sample_100` folder. Reuse its cache and original seed order:

```bash
python sample_dataset.py --count 1100 --rows-per-destination 200 --output data/graphs/sample_1100
python validate_dataset.py --data data/graphs/sample_1100 --retained-from data/graphs/sample_100
```

This is 1,100 total source articles, not 1,100 extra. Cached BFS replay preserves the original 100 when using the same original cache, seeds, and sampler. Check the retained page IDs when using a different cache or editing sampling settings. An interrupted collection can be rerun with the same arguments; complete article records are reused. Different seed data and live edits can change a fresh sample.

The larger export samples at most 200 valid current→candidate edges per destination before writing rows, limiting output to at most 220,000 rows rather than writing every edge for every destination. Labels remain exact distances in the larger graph. Distances are recomputed by reverse BFS per destination, rather than searching separately for every source/destination pair. Destinations are split 80/10/10 before examples are generated.

First test whether the **frozen 100-article model** transfers to new articles:

```bash
python compare_search.py --data data/graphs/sample_1100 --checkpoint runs/model_100/best.pt --novel-destinations --output runs/transfer_100_to_1100.json --queries 100 --budget 200
```

Here destinations are outside all 100 original article IDs. Source articles can be old or new. This is transfer to an expanded, overlapping graph, not evaluation on an entirely independent Wikipedia sample. Reference optimal distances are recomputed on the larger graph; old distances can be too long after new articles add shortcuts.

Then train a **separate model on the larger graph**:

```bash
python train_model.py --data data/graphs/sample_1100 --output runs/model_1100 --epochs 25
python compare_search.py --data data/graphs/sample_1100 --checkpoint runs/model_1100/best.pt --output runs/comparison_1100.json --queries 100 --budget 200
```

Keep `model_100` and `model_1100` separate. Each within-graph model uses its own held-out destination split. The transfer and larger-model reports may use different query sets, so their aggregate numbers are not a controlled head-to-head comparison between model sizes. Within each report, BFS, title similarity, and neural-guided search use exactly the same queries and expansion budget.

## Read the comparison

Generate the recorded training and comparison charts without retraining:

```bash
python -m pip install -r requirements-plot.txt
python plot_results.py
```

PNG and PDF charts are written to `runs/figures/`, with a CSV summary. `training_curves` separates mean training Smooth L1 loss from validation MAE: they are different quantities, so a numerical gap between them is not an overfitting measurement. Training loss is averaged over batches while weights are being updated; validation MAE is evaluated at the end of the epoch. The selected-epoch markers identify the lowest validation MAE. The two model curves describe different graph tasks and held-out sets, so their relative heights do not establish a training-size effect.

`loss_function` plots the fixed Smooth L1 penalty with beta 1; the loss shape is chosen in advance, not learned. For residual e = predicted distance minus true distance, the penalty is 0.5 e² when |e| < 1, and |e| - 0.5 otherwise.

`search_comparison` shows expansion work, optimal/longer/failed routes, offline latency, and success versus expansion budget. The budget curves are derived from each recorded deterministic search's stopping expansion count for budgets up to the original 200; they do not extrapolate beyond measured searches. These plots use the paired held-out queries in `comparison_1100.json` and do not rerun or tune models using test outcomes.

Define a claim about intelligent guidance operationally: higher success under the same work budget, or less work for an agreed route-quality requirement. Also report shortest-route correctness and wall-clock cost. Neither lower training loss nor fewer expansions alone establishes a better search system. For stronger evidence, predefine query sets, include unsuccessful searches, compare with the non-neural heuristic, repeat model training across seeds, and use paired uncertainty intervals. A claim about API fetching additionally requires measuring HTTP calls on equivalent cache states; these offline figures do not provide that evidence.

The comparison methods are ordinary BFS, greedy best-first ordered by title similarity, and greedy best-first ordered by predicted remaining distance. All searches use a frontier and prevent revisiting discovered nodes. Neural and text priorities need not return shortest routes; they are not admissible A* heuristics.

Reference BFS distances select graph-reachable evaluation queries at least two clicks long and measure route quality. They never guide the evaluated neural or text searches. This experiment does not evaluate unreachable-goal detection. Failures under the search budget remain in the success-rate denominator.

- `success_rate`: fraction reaching the goal under the expansion budget.
- `optimal_route_rate_all_queries`: fraction returning the exact minimum click count.
- `mean_extra_clicks_successes_only`: excess clicks among successful routes; read alongside success rate.
- `mean_expanded_all_queries`: average articles whose outgoing neighbors are examined, including failed searches.
- `median_seconds_all_queries`: per-query search time, including neural inference and tensor construction.

Shared title-feature preparation is timed separately. Checkpoint loading and reference answer calculation are outside per-query search timing. On small saved graphs, BFS may be much faster even when neural guidance expands fewer articles. Current timing is a single local pass, not a statistically robust latency benchmark.

**This is offline search: no API requests occur during comparison.** An expansion count is not an HTTP-request count. Live fetching would need equivalent initial caches, page/link access, request budgets, and accounting for pagination and feature fetches. No live-fetch speedup is claimed.

## Files and storage

To compute the exact reference average over **every ordered pair** in the expanded graph:

```bash
python analyze_all_pairs.py --save-distances
```

This runs BFS from each source and excludes source-equals-target pairs. `runs/all_pairs_1100/summary.json` records the weighted average over all reachable pairs, histogram, maximum finite distance, and unreachable count. `per_source.csv` records source-specific statistics. Optional `pair_distances.csv.gz` contains every distinct ordered pair; an empty distance means no route within the saved graph, not zero clicks. Unreachable pairs are excluded from a finite mean and counted separately.

Initial exhaustive result: 1,208,900 distinct ordered pairs, 1,207,801 reachable and 1,099 unreachable. The sum of reachable shortest distances is 2,904,126, giving a mean of **2.4044739158 clicks**, median 2, 95th percentile 3, and maximum finite distance 6. This is different from the earlier 2.47-click average over 100 selected evaluation queries, which excluded direct one-click routes.

Training scripts and dependency pins belong in Git. Collected data remains in `data/graphs/`, resumable source records in `data/cache/`, and models/results in `runs/`; these generated directories are ignored by Git. SQL is not required. Reports identify the graph with its SHA-256 content ID.

See the [PyTorch quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) for tensors, neural modules, DataLoader, losses, backpropagation, and optimizers.
