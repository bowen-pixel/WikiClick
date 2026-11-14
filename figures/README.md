# Figure catalogue

Identifiers below are used in the root Markdown files. Images were copied without changing pixels or smoothing. Original labels inside images are retained. The machine-readable catalogue records source checksums and the matching public code file. Historical reproduction drivers and full experiment logs stay local.

| ID | Figure | Meaning |
|---|---|---|
| LC01 | [LC01_distance_baseline](learning_curves/LC01_distance_baseline.png) | Original distance regression; dataset panels retain their original labels |
| LC02 | [LC02_sha256_vs_embeddinggemma_1000](learning_curves/LC02_sha256_vs_embeddinggemma_1000.png) | Matched title-feature comparison, 1,000 articles; validation/train MAE |
| LC03 | [LC03_next_link_validation_1000](learning_curves/LC03_next_link_validation_1000.png) | Next-link ranking, 1,000 articles; original per-seed validation histories |
| LF01 | [LF01_smooth_l1_illustration](loss_functions/LF01_smooth_l1_illustration.png) | Mathematical Smooth L1 shape; NOT a measured optimization landscape |
| NV01 | [NV01_next_link_goal_success_1000](navigation/NV01_next_link_goal_success_1000.png) | Held-out goal completion, 1,000-node graph |
| NV02 | [NV02_next_link_shortest_routes_1000](navigation/NV02_next_link_shortest_routes_1000.png) | Held-out shortest routes, 1,000-node graph |
| NV03 | [NV03_branching_goal_success_million](navigation/NV03_branching_goal_success_million.png) | Historical hashed text/category model; NOT the live EmbeddingGemma model |
| NV04 | [NV04_branching_shortest_routes_million](navigation/NV04_branching_shortest_routes_million.png) | Historical million-node graph; shortest-route rate under matched budgets |
| EV01 | [EV01_architecture_search_100_journeys](evolutionary/EV01_architecture_search_100_journeys.png) | 12 candidates per arm, 100 held-out journeys; evolution vs random control |
| EV02 | [EV02_selected_residual_architecture](evolutionary/EV02_selected_residual_architecture.png) | Selected residual architecture schematic; not a measured learning curve |
| RW01 | [RW01_offline_navigation_scaling](random_walk/RW01_offline_navigation_scaling.png) | Offline 10A scaling; selected corpus D1 shortest states, not live random-walk deployment |
| RW02 | [RW02_offline_ranking_curves](random_walk/RW02_offline_ranking_curves.png) | Offline 10A neural ranking curves; all experimental controls remain local |

[Source checksums and code map](statistics/figure_catalog.json). [1,000-node statistics](statistics/next_link_1000.json), [title features](statistics/title_feature_comparison.json), [evolution](statistics/evolution_100.json), [offline scaling](statistics/navigation_scaling_10a.json). These experiments use different graphs, inputs and budgets; their percentages are not directly interchangeable.
