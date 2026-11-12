# Saved PyTorch models

| Folder | Contents |
|---|---|
| `raw/model_100/` | Original 100-article distance checkpoint and metrics |
| `raw/model_1100/` | Original 1,100-article distance checkpoint and metrics |
| `raw/model_large_trained/` | Larger-graph fine-tuned distance checkpoints, progress, metrics |
| `next_link/ranking_1000/` | Three seeds of raw, title-ranker, semantic-ranker, continuation, and correction checkpoints; cached feature tensor; histories and matched routes |
| `next_link/ranking_million/` | Completed million-scale experiment, three seeds, selection records, matched route evaluations, workflow metadata |
| `evolution/evolution_100/` | Selected evolution/random models, candidate histories, parentage, 100-query evaluation |

BFS is an algorithm and has no `.pt` weights. Model implementation code is under `codes/models/`; graph and label data are under `data/`. Complete experiment metadata stays beside each checkpoint so its provenance and evaluation are not separated.

Use `wikiclick.py next-link --model raw|title_rank|semantic_rank|continued|corrected|evolutionary|random_nas` to select a saved small-graph policy. The default is unchanged by the file cleanup. Larger models use the separate documented million-scale workflow.
