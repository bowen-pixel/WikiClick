# 1,000-article next-link ranking experiment

Measured local experiment; no guaranteed improvement is assumed.

Graph: **1,000 articles, 54,404 directed links**. Graph ID: `sha256:c9a5fcd48d1d387efd844aa7f4bac1b9eb0f1f876c0918024d15aa53461a10f7`.
The 1,000 nodes are a deterministic induced subset of the existing 1,100-article collection, not 1,000 new downloads.

Fetched current introduction text for 999/1000 articles. Retained 833 category hyperedges, covering 860 articles. Text collection is later than the hyperlink collection. Empty introductions use zero content features. Sources/revisions are cached.

## Protocol

Destination-disjoint splits; 20-click navigation with no revisits, direct target check shared by neural methods. BFS searches up to all nodes and is an optimality reference, not an equal-work navigation policy. Features precomputed for all methods; no live network calls during evaluation.

Destinations: 800 train / 100 validation / 100 test. Decisions: {'train': 8000, 'validation': 1000, 'test': 1000}. Seeds: [207, 208, 209].
Each decision contains all outgoing neighbors and all equally optimal labels. Labels are exact in this saved graph, not the full web. Train/validation/test destinations do not overlap. Article text and category membership across the fixed graph are visible as unsupervised transductive features.

Raw baseline: original title-only distance architecture (34,945 parameters), trained from random weights on at most 60,000 unique neighbor/destination pairs from the training decisions, Smooth L1 loss. Test/validation targets are never used in optimizer updates.
Ranking: 10 extra epochs from the raw checkpoint. Title ranker retains 34,945 parameters; semantic ranker adds a 16,449-parameter residual head using intro hashes and fixed category-hypergraph pooling (51,394 total). This is not a pretrained semantic encoder or a learned hypergraph encoder.
Correction: 5 extra epochs, training mistakes sampled with weight 4. Ordinary continuation receives the same number of updates from the same semantic checkpoint. Both select checkpoints by validation cross-entropy, including the option of retaining the starting checkpoint. Final test labels are opened only after every seed/condition is trained.

## Actual held-out results

Percentages are means across seeds; each seed uses the identical test decisions and journeys.

| Model | Optimal next link | Goal reached | Optimal route | Median ms (mean across seeds) |
|---|---:|---:|---:|---:|
| Raw distance model | 70.87% | 91.53% | 51.17% | 0.478 |
| Title next-link ranker | 79.40% | 98.50% | 66.17% | 0.365 |
| Text + hypergraph ranker | 80.67% | 98.57% | 68.13% | 0.543 |
| Ordinary continuation | 80.67% | 98.57% | 68.13% | 0.556 |
| Correction (prior model retained) | 80.67% | 98.57% | 68.13% | 0.549 |

BFS optimal reference: 1000/1000 reached; 1000 optimal. BFS has a different work limit and is not presented as a matched navigation policy.

## Paired route-success gains

- Full ranking pipeline − raw model: +7.03 percentage points; conditional 95% interval [+3.23, +10.80].
- Correction − ordinary continuation: +0.00 percentage points; conditional 95% interval [+0.00, +0.00].
- Semantic context − title ranking: +0.07 percentage points; conditional 95% interval [-0.93, +1.07].

## Was the correction retained?

Epoch 0 means validation retained the original semantic checkpoint; correction updates were attempted but not deployed.

| Seed | Continuation selected epoch | Correction selected epoch | Correction selected-weight change (L2) |
|---:|---:|---:|---:|
| 207 | 0 | 0 | 0.000000 |
| 208 | 0 | 0 | 0.000000 |
| 209 | 0 | 0 | 0.000000 |

Intervals cluster by destination, averaging seeds within each paired query; they do not establish generalization beyond this graph. We did not retune using test outcomes. Correction need not beat ordinary continuation, and semantic context need not help.

## Local deployment

`rank_navigate.py` loads saved weights and cached features locally. It does not fetch live pages, update weights during inference, modify the hyperlink graph, or use true distances to choose moves. Seed 207 is the declared default, not selected using test performance.

```bash
python -m codes.models.next_link.rank_navigate
python -m codes.models.next_link.rank_navigate --model raw --source 'Graph theory' --target 'Mathematics'
```

Only the saved 1,000 titles are supported. The corrected checkpoint is an experimental alternative, not automatically a better default. Use `--model continued`, `--model semantic_rank`, or `--model title_rank` to compare.

## Reproduce

```bash
python -m codes.models.next_link.next_link_experiment prepare
python -m codes.models.next_link.next_link_experiment fetch
python -m codes.models.next_link.next_link_experiment run
python -m codes.plotting.plot_next_link
```

Existing source caches are reused. Training commands restart rather than resume optimizer sessions. No earlier million-node artifacts or checkpoints were overwritten. Self-correction here means offline supervised hard-example resampling, not autonomous fact verification, RL, Coconut, or test-time training.
