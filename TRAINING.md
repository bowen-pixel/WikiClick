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

Training scripts and dependency pins belong in Git. Collected data remains in `data/graphs/`, resumable source records in `data/cache/`, and models/results in `runs/`; these generated directories are ignored by Git. SQL is not required. Reports identify the graph with its SHA-256 content ID.

See the [PyTorch quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) for tensors, neural modules, DataLoader, losses, backpropagation, and optimizers.
