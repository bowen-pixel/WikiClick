# Timeline

Dates below come from Git commit dates, the existing AI-use record, and this chat. Experiment numbers are development stages, not dates. Where original timing/idea attribution is incomplete, it is not reconstructed from a filename.

| Date / stage | Change | Achieved and still limited |
|---|---|---|
| 2026-11-04 | Initial directed-graph/BFS project plan and documentation | Plan only; no trained model at this point |
| 2026-11-05 | Wikipedia API and interactive source/target input | Small-domain canonical link extraction; exact shortest paths inside the saved graph |
| 2026-11-05 | 100-article collection and then 1,000 additional articles; PyTorch training | Saved graph-relative supervision and distance models; not all-Wikipedia search |
| 2026-11-05 onward | Curves, losses and search comparisons | Measured curves separated from mathematical loss illustrations (**LC01/LF01**) |
| 2026-11-10 | Larger graph/million-node workflow recorded in Git | Sparse graph evaluation with sampled supervision; collecting nodes is not training on every node |
| 2026-11-11 | Next-link ranking and offline correction; experiment reviews and package organization | Ranking improved recorded small-graph goal completion; correction retained the previous checkpoint (**LC03/NV01/NV02**) |
| 2026-11-11 | Evolutionary architecture pilot with random-search control | 100 shared test journeys; no established overall evolutionary gain (**EV01/EV02**) |
| 2026-11-12 | Regularization/title-feature inspection and learning plan | SHA256 and frozen semantic title comparisons; MAE gains do not alone prove routing gains (**LC02**) |
| Later saved stages 5–10; precise chronology not reconstructed | Pairwise ranking, contextual relevance, RFBC/walk supervision, graph transfer, adaptive and evolutionary priority controls, residual scaling | Completed local experiment artifacts retained. Public mathematical primitives and all recoverable training corpora are preserved; full experiment drivers/logs stay local (**RW01/RW02**) |
| 2026-11-13–14 chat work | Live API-backed URL navigation | On-demand links/features and semantic best-first; example routes checked, global shortest-path frequency not benchmarked |
| 2026-11-14 chat work | Two-link probes and interactive URL prompts | Verified Newton → United States customary units → Apple; numbered URLs and click count; CPU encoding remains expensive on large frontiers |
| 2026-11-14 | FC Barcelona → Gao Qiu latency inspection | Measured 1,047 outgoing candidates, five initial API queries, about 0.9 seconds in one check; identified subsequent API batches and CPU encoding as additional work |
| 2026-11-14 | Public-core refactor | `code/`, consolidated `training_data/`, labelled `figures/`, deployment/foundations/timeline/AI docs; local tests/experiments preserved and excluded from the next public snapshot |

Future milestones are not completed work: persistent caches, candidate preselection, phase timing, random-walk-guided live fetching, and a fresh held-out live route/latency benchmark. This refactor does not retrain models, change historical metrics or publish a hosted website.
