# Initial PyTorch experiment results

Run date: November 5, 2026 (America/Toronto). CPU; PyTorch 2.11.0; seed 207. These are initial single-run results, not a general performance claim.

## Data and validation

The expanded graph retains all original 100 page IDs and adds 1,000 articles: 1,100 nodes, 63,966 unit-weight directed edges. All 220,000 capped training rows were checked against independent forward BFS distances. Destination partitions are disjoint. The original 100-article dataset remains unchanged.

## Distance prediction

| Model graph | Selected epoch | Test MAE (clicks) | Constant baseline MAE | Unique training pairs used |
| --- | ---: | ---: | ---: | ---: |
| model_100 | 13 | 0.593 | 0.853 | 7,446 |
| model_1100 | 8 | 0.414 | 0.531 | 30,000 |

The two models have different graph-relative labels and held-out destination sets. Do not interpret this table as a controlled comparison of training-set sizes. The baseline is the mean training distance, evaluated on the same selected test pairs as its corresponding model. Null labels are excluded from regression.

## Search experiments

Within each experiment all methods receive the same reachable source/target queries (minimum two clicks) and a 200-expansion budget. BFS stops on first goal discovery; guided methods use the same stopping rule. Failures count against success and optimal-route rates. Timing includes per-query inference, but shared title preparation is reported separately in JSON. No API calls occur during these offline searches.

### 100-article graph: held-out destinations

| Method | Reached / queries | Optimal routes / queries | Mean expansions | Extra clicks, successes only | Median search ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| bfs | 100/100 | 100/100 | 13.62 | 0.00 | 0.015 |
| text | 100/100 | 75/100 | 30.01 | 0.42 | 0.178 |
| neural | 100/100 | 96/100 | 5.26 | 0.05 | 0.168 |

Detailed local report: `runs/comparison_100.json`. Graph ID: `sha256:c1e952de422d26c5ee4f34313ed8def15923008aae9bfa4e77e957fc0b4eefab`.

### Frozen 100-article model on expanded graph: new destinations

| Method | Reached / queries | Optimal routes / queries | Mean expansions | Extra clicks, successes only | Median search ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| bfs | 91/100 | 91/100 | 39.89 | 0.00 | 0.089 |
| text | 95/100 | 31/100 | 25.10 | 1.76 | 0.817 |
| neural | 98/100 | 47/100 | 20.08 | 1.44 | 1.641 |

Detailed local report: `runs/transfer_100_to_1100.json`. Graph ID: `sha256:1247d68432e26cd97b8bbded6d7e9a1e7b3207f1f93b993af05bcc3bb80ca712`.

### 1,100-article model on expanded graph: held-out destinations

| Method | Reached / queries | Optimal routes / queries | Mean expansions | Extra clicks, successes only | Median search ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| bfs | 97/100 | 97/100 | 37.23 | 0.00 | 0.071 |
| text | 96/100 | 35/100 | 23.66 | 1.58 | 0.632 |
| neural | 100/100 | 69/100 | 8.25 | 0.37 | 1.059 |

Detailed local report: `runs/comparison_1100.json`. Graph ID: `sha256:1247d68432e26cd97b8bbded6d7e9a1e7b3207f1f93b993af05bcc3bb80ca712`.

The transfer and retrained-model experiments use different query sets. Compare the methods within each table; a direct comparison between models requires a shared query set held out from both. BFS failures here are budget exhaustion, not incorrect optimal distances. Neural guidance explores fewer articles in these runs but can choose longer routes and incurs inference overhead. Title-only hashed features, correlated examples, topic-biased sampling, and a shared fixed graph limit the conclusions. A fair live-fetch benchmark is still future work.
