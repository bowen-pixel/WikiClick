# Actual supervised training after adding a second million articles

Started November 10, 2026 (America/Toronto). This run **updates network weights** using exact labels from the expanded graph. It is separate from the earlier million-page frozen-model transfer test.

## Data expansion

The complete recent Wikimedia snapshot is already downloaded. No new network download is needed to select another million distinct articles from it. The importer retains all **1,001,100** previous article IDs and adds **1,000,000** more, targeting **2,001,100 nodes**, with fresh canonical hyperlinks, unit edge costs, deduplication, and redirect resolution from the same pinned snapshot. Data and the existing models remain in their original folders; the expanded graph is written to `data/graphs/fresh_two_million/`.

## Supervised examples and leakage control

- Select destinations only from the newly added million articles: **200 training**, **25 validation**, and **25 test** destinations, with no destination overlap.
- Compute exact reverse-graph shortest distances for these 250 destinations. Per-destination cached labels permit restarting this stage.
- Sample up to **2,048 candidate articles per destination**, giving at most **512,000 exact distance pairs** in total. Candidate articles can come from the entire expanded graph. This is sampled supervised training on the graph, not training on every article's full text or labeling every possible pair.
- Unreachable vertices are omitted from numeric regression rather than assigned distance zero.
- Input features remain fixed title hashes. Their faster training-time construction is tested for exact equivalence to the existing feature representation.
- Only pair identities are read for unsupervised feature caching. Test labels are loaded by the trainer only after checkpoint selection.
- The evaluation query file is drawn from held-out test destinations and is reused for the frozen and newly trained models. Original previously evaluated destinations are outside this new destination pool.

## Real weight updates

Initialize from the existing `models/raw/model_1100/best.pt`, then run **25 fine-tuning epochs** using Adam (learning rate 0.0003), Smooth L1 loss, batches of 256, and seed 208. The optimizer performs backpropagation and updates parameters on expanded-graph training examples. The initial validation error is recorded, validation selects the best checkpoint (including the option to retain the initial model if no trained epoch improves it), and the selected model receives a final test evaluation.

The trainer saves:

- `models/raw/model_large_trained/last.pt`: the latest epoch's actual trained weights and optimizer state.
- `models/raw/model_large_trained/best.pt`: the validation-selected checkpoint.
- `models/raw/model_large_trained/progress.json`: current epoch, optimizer update count, and losses.
- `models/raw/model_large_trained/metrics.json`: initial/final validation metrics, test MAE, training history, update count, and L2 difference between selected and initial parameters.

This is fine-tuning, not training a language model from scratch. Graph hyperlink costs stay at 1; the updated parameters belong to the neural distance predictor. Training does not guarantee that title-only guidance will outperform the text heuristic or BFS.

## Paired comparison and plots

After training, evaluate both the old frozen checkpoint and new checkpoint on the **same 100 held-out queries in the same expanded graph**. BFS and text similarity are reference methods. Each run uses 200-expansion and 10-second limits, with cold feature costs included consistently.

Results go to `reviews/results/large_frozen_comparison.json` and `reviews/results/large_trained_comparison.json`. Search diagrams are saved separately in `figures/raw/large_frozen/` and `figures/raw/large_trained/`. The trained directory additionally contains its own training/validation curves, based on actual recorded fine-tuning epochs. No pilot loss history is reused as if it were large-graph training.

## Watch the job

```bash
cat reviews/results/large_training_status.json
tail -f runs/large_training_job.log
```

During graph expansion, inspect `runs/large_training_import.log`. During label generation inspect `data/graphs/large_supervised/progress.json`. During training inspect `models/raw/model_large_trained/progress.json`. `model_weights_updated` is false until training has actually run; final results exist only when the job reports completion.

The current worker waits for the already-running expansion. Do not launch duplicate workers. To restart an inactive/failed workflow after inspecting its logs:

```bash
python -m codes.workflows.run_large_training
```

Existing graph, label, and title-feature stage checkpoints are reused. The current trainer keeps epoch artifacts but restarts fine-tuning from the initial checkpoint if its training command is rerun; it does not yet automatically resume an interrupted optimizer session.

Once ready:

```bash
open figures/raw/large_trained/training_curves.png
open figures/raw/large_trained/search_comparison.png
```

The run is local and can proceed without Wi-Fi after the existing bulk files have been verified. The computer must stay running. The implementation passed **30 tests**, including a small end-to-end example verifying new-goal splits, exact labels, changed parameters, and identical comparison queries before this real job was launched.

## Completed supervised training run

The expanded graph contains **2,001,100 articles and 56,350,501 edges**, retaining all previous 1,001,100 article IDs. The supervised stage finished 25 epochs and **39,625 optimizer updates**. Actual example counts were **405,522 training**, **51,200 validation**, and **49,158 test** pairs; some sampled goals had fewer than 2,048 reachable candidates. The selected epoch was **1**, with selected-weight L2 change **7.25848** relative to the old checkpoint, confirming that deployed-selected weights were updated rather than merely reused.

Initial validation MAE was **2.76868**, selected validation MAE **1.06644**, and test MAE **0.90289 clicks**. The constant training-mean distance baseline achieved test MAE **0.57463**, so the neural predictor is not the best distance estimator under this held-out test. Distance-prediction error and route-search success are different objectives; the paired routing comparison must be inspected separately.

The trained checkpoint and actual learning curves are available. At this entry the paired old-versus-trained routing benchmark was still in progress. Do not describe final search success or superiority until its measured reports exist.
