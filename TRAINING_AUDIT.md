# WikiClick experiment audit

Cross-size tests differ; only the two-million frozen/fine-tuned comparison is matched.

Articles are graph nodes, not training examples. Labels are sampled candidate/destination distances.

| Graph articles | Training | Training pairs | Neural test MAE | Constant MAE | Neural routes found | BFS routes found |
|---:|---|---:|---:|---:|---:|---:|
| 100 | from scratch | 7,446 | 0.593 | 0.853 | 100/100 | 100/100 |
| 1,100 | from scratch | 30,000 | 0.414 | 0.531 | 100/100 | 97/100 |
| 1,001,100 | none: frozen transfer | 0 | Not measured | Not measured | 7/100 | 3/100 |
| 2,001,100 | fine-tuning | 405,522 | 0.903 | 0.575 | 0/100 | 0/100 |

No saved training run exists for exactly 10, 1,000 or 1,000,000 articles.

## Matched two-million comparison

Same 49,158 held-out distance pairs:
- Frozen 1,100-article model: 2.502322 clicks MAE.
- Fine-tuned model: 0.902893 clicks MAE.
- Constant train-mean predictor: 0.574634 clicks MAE.

All four routing approaches reached 0/100 goals on the shared test queries within 200 expansions and 10 seconds. All stopped at the expansion budget. These pairs have known paths; this is a budget floor, not evidence that paths do not exist.

The constant distance baseline is not a routing policy; beating it on MAE is a useful sanity check, but routing quality needs its own test.

## Interpretation

Small runs used supervised training from random initialization. The million run was frozen inference only. The two-million run fine-tuned the 1,100 model with real optimizer updates. No test-time training has been implemented.

The model uses hashed article titles in a small MLP, not article text or graph structure. Falling training loss alone does not establish generalization or better routing. Lower validation error relative to the old model did not beat a constant on the test split or improve routing success.

## Next controlled experiment

Keep one graph and destination-disjoint validation/test sets fixed. Vary nested training-pair counts, train multiple seeds, and compare against the train-mean distance baseline and BFS. Report success, optimal-route rate, expansions and total latency across larger search budgets. Choose budgets on validation queries first; reserve new test queries after tuning. Test graph/text features or a neighbor-ranking objective before collecting more pages. Any test-time updates require a separately defined objective and an evaluation that includes update costs and excludes true test-distance labels.

Rebuild these outputs with `.venv/bin/python plot_training_audit.py` from the repository directory.
