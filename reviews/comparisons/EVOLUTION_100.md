# Evolutionary architecture-search pilot

Measured November 11, 2026 (America/Toronto). A completed local experiment, not an established improvement.

## Diagrams

The supplied screenshot appears to show particle-swarm optimization of a sphere function: contours and a 3D objective surface with moving candidate positions. It is not a picture of neurons learning. The `w`, `c1`, and `c2` labels are characteristic of particle-swarm optimization. The exact software used for the screenshot is unknown.

Our figures use recorded measurements and Matplotlib; `FuncAnimation` and Pillow produce the GIF. PNG, PDF, and SVG exports are under `figures/evolution/evolution_100/`:

- `search_dashboard`: validation search progress, training/validation losses, final test counts.
- `selected_architecture`: original frozen ranker and selected refinement branch.
- `architecture_search_map`: evaluated architectures, parent mutations, validation scores.
- `evolution.gif`: the architecture-search map animated over candidate evaluations.

These are discrete architecture measurements; no smooth loss surface is invented between unevaluated models. Training loss averages predictions during updates; validation loss measures the model after each epoch.

## Model and search

Reused the existing 1,000-article graph (54,404 directed links), cached title/intro/category features, and `models/next_link/ranking_1000/seed_207/semantic_rank.pt`. The base has 51,394 parameters and remains frozen. Each candidate adds a scoring branch initialized to output zero, preserving initial predictions.

There are 36 possible architectures: inputs (title pairs, text/category pairs, both), hidden width (16/32/64), hidden depth (1/2), and activation (ReLU/GELU). This is a bounded residual-adapter architecture search, not a reproduction of Evolved Transformer, NEAT, or full-network topology evolution.

Population size 4; 3 generations; 12 unique candidates, each trained for 3 epochs on all 8,000 training decisions. Later generations use two-member tournament parent selection, a one-gene mutation, and elitist survivor selection. An unseen random immigrant is used if all neighboring genotypes have been evaluated. Children inherit architecture choices, not trained adapter weights. Every candidate starts from the same frozen base.

Random search evaluates 12 candidates for the same 3-epoch budget. Both arms start with the same four random candidates. Matching genotypes receive matching training seeds and data order. Equal candidate/update budgets do not imply equal FLOPs. The unchanged baseline is eligible for rollback.

Checkpoint and architecture selection prioritize: validation journey completion, shortest-route count, lower ranking cross-entropy on 1,000 validation decisions, then smaller parameter count. There are 100 validation journeys, one per validation destination. This rule can select a checkpoint with worse ranking loss if route results improve. Test results never select checkpoints.

Selected evolution branch: both inputs, width 64, two hidden layers, ReLU, epoch 2; 121,219 total parameters (69,825 trainable). Selected random-search branch: both inputs, width 16, one hidden layer, ReLU, epoch 3; 67,811 total parameters. Evolution selected `evolution_09`; random search retained initial-population candidate `random_01`.

Both selected models reached 100/100 validation goals with 69 shortest routes; the original ranker reached 99/100 with 68 shortest routes. These are selection-set measurements.

## Final evaluation on 100 sample points

“100 sample points” means 100 complete navigation journeys: one reachable source/destination query per held-out test destination, with exact distance at least two. The 800/100/100 train/validation/test destination partitions remain disjoint. Test pairs exclude every pair in the earlier evaluation suite. Test queries and labels are generated after both searches finish.

These destinations were evaluated in the earlier experiment. New source pairs do not make this a pristine benchmark. Unsupervised features from all graph articles are visible (transductive evaluation). This describes one graph and one search seed, not generalization across Wikipedia.

All models share a 20-click limit, no revisits, and direct-goal check. No network calls, label-based move selection, or test-time training occur. Exact BFS supplies reference distances, not an equal-budget navigation competitor. Returned routes were checked for valid directed edges, no revisits, and consistency with exact distances.

| Model | Goals reached | Shortest routes | Mean extra clicks, successes only | Median query time |
|---|---:|---:|---:|---:|
| Existing ranker | 99/100 | 76/100 | 0.788 | 0.617 ms |
| Evolution selected | 99/100 | 75/100 | 0.505 | 1.221 ms |
| Random-search selected | 99/100 | 74/100 | 0.636 | 1.137 ms |

Evolution changed one query from nonoptimal to optimal and two in the opposite direction. The paired optimal-route difference is -1 percentage point, with a conditional bootstrap 95% interval of [-4, +2]. All models succeeded and failed on the same queries. The zero bootstrap interval for completion differences reflects identical observed outcomes, not proof of future equivalence. Latency includes per-query inference; cached feature preparation, training, and search are excluded.

Evolution reduced average detour length among successful journeys but did not improve completion or shortest-route count, and added inference overhead. This pilot does not establish superiority over the existing model or random search. No search was retuned after these outcomes. The navigation default remains unchanged.

Search and evaluation took 216.66 seconds, excluding plotting. Checkpoints, histories, parentage, query lists, per-query routes, and metrics are saved in `models/evolution/evolution_100/`. All 52 project tests passed, including mutation uniqueness, frozen weights, source-pair exclusion, and checkpoint reload/rollback.

## Reproduce and use

From this repository directory:

```bash
.venv/bin/python -m codes.models.evolution.evolutionary_search --output models/evolution/evolution_100_repeat
.venv/bin/python -m codes.plotting.plot_evolutionary_search --run models/evolution/evolution_100_repeat
.venv/bin/python -m codes.models.next_link.rank_navigate --model evolutionary
.venv/bin/python -m codes.models.next_link.rank_navigate --model random_nas
```

Navigation uses the completed original pilot by default. Add `--search-run models/evolution/evolution_100_repeat` for a repeat. Only the saved 1,000 titles are supported. Search refuses to overwrite existing output directories. Earlier models, graphs, cached collections, and the paused million-article workflow are preserved.

Implementation: `evolutionary_search.py`; plotting: `plot_evolutionary_search.py`; explicit inference options: `rank_navigate.py`.

Plotting references: [Matplotlib animation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html), [PySwarms sphere example](https://pyswarms.readthedocs.io/en/latest/examples/tutorials/basic_optimization.html).
