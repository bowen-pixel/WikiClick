# Figures

Figures are grouped by the method that produced them. PNG is convenient for viewing; PDF/SVG exports are available where generated. The charts show measured outcomes unless explicitly labeled as a schematic or mathematical loss illustration.

| Method | Learning/training | Routing/search |
|---|---|---|
| Original distance model | [Training curves](raw/baseline/training_curves.png) | [Search comparison](raw/baseline/search_comparison.png) |
| Larger distance-model fine-tuning | [Fine-tuning curves](raw/training_audit/fine_tuning_curves.png) | [Matched search outcomes](raw/training_audit/matched_search_outcomes.png) |
| Next-link ranking, 1,000 articles | [Validation curves](next_link/ranking_1000/validation_curves.png) | [Goal completion](next_link/ranking_1000/route_success.png), [shortest routes](next_link/ranking_1000/optimal_routes.png), [paired gains](next_link/ranking_1000/paired_gains.png) |
| Million-scale next-link ranking | Histories remain in the [model bundles](../models/next_link/ranking_million/) | [Branching completion](next_link/ranking_million/branching_success.png), [shortest routes](next_link/ranking_million/branching_optimal.png) |
| Evolutionary refinement | [Search and training dashboard](evolution/evolution_100/search_dashboard.png) | [Architecture](evolution/evolution_100/selected_architecture.png), [search animation](evolution/evolution_100/evolution.gif) |

Additional raw-model comparisons live in `raw/large_frozen/`, `raw/large_trained/`, and `raw/fresh_million/`. Earlier differing chart versions are retained in `history/`.

From the project root, rebuild without retraining:

```bash
.venv/bin/python wikiclick.py plot-raw
.venv/bin/python wikiclick.py plot-next-link
.venv/bin/python wikiclick.py plot-million
.venv/bin/python wikiclick.py plot-evolution
```

`plot-audit` also recomputes the saved models' distance errors on cached labels. The plotting commands write to this figure tree and their corresponding comparison reviews; they do not choose new model weights.

Smoother curves are not automatically better. Validation loss and navigation results answer different questions. We retain the measured curves rather than smoothing away differences or presenting a fabricated loss surface.
