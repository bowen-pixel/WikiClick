# WikiClick

Find hyperlink routes between Wikipedia articles. BFS provides the exact shortest-path reference within a saved graph; PyTorch models score links to guide navigation and search.

The project is consolidated here. There is no longer a second `WikiClick/` project inside this folder. Start with this page and the [comparison overview](reviews/README.md).

Latest inspection: [Model structure and next learning experiments](reviews/planning/LEARNING_IMPROVEMENT_PLAN.md), covering semantic title features, checkpoint selection, hidden layers, epoch budgets, and grouped splits.

## Where things live

| Folder | Contents |
|---|---|
| `codes/api/` | Wikipedia API access, JSON collection/validation, dump import, cached semantic features |
| `codes/search/` | CLI article input, graph loading, BFS, distance analysis, search comparisons |
| `codes/models/raw/` | Original PyTorch distance model and training |
| `codes/models/next_link/` | Next-link ranking, text/category refinement, correction experiment, inference |
| `codes/models/evolution/` | Evolutionary architecture-search experiment |
| `codes/workflows/` | Larger collection/training workflows and status commands |
| `codes/plotting/` | Rebuild charts from saved measurements |
| `data/` | Graph JSON, training JSON/JSONL, sparse arrays, source caches, raw dumps |
| `models/` | Saved `.pt` weights, feature tensors, training histories, model-specific results and routes |
| `reviews/` | Approach comparisons, guides, planning, AI-use record, machine-readable comparison reports |
| `figures/` | Charts and animations grouped by raw, next-link, and evolutionary methods |
| `logs/` | Collection/training logs and regeneratable runtime caches |
| `tests/` | API, graph, learning, workflow, and regression tests |
| `requirements/` | Dependency sets: base, dev, ML, plots, scale |
| `archive/` | Differing older files and preserved inner-repository history |

Model bundles retain histories, query lists, and results beside their weights so the evidence remains tied to each model. BFS has no learned checkpoint. JSON remains under the dataset or experiment it describes; files were not mixed into one large JSON folder.

## Run it

Use Python 3.13. The existing `.venv` has been consolidated and retained. For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/scale.txt
```

One launcher replaces the scattered top-level Python scripts:

```bash
.venv/bin/python wikiclick.py --help
.venv/bin/python wikiclick.py bfs --data data/graphs/sample_100/articles.json
.venv/bin/python wikiclick.py next-link
.venv/bin/python wikiclick.py next-link --model raw
.venv/bin/python wikiclick.py next-link --model title_rank
.venv/bin/python wikiclick.py next-link --model semantic_rank
.venv/bin/python wikiclick.py next-link --model evolutionary
.venv/bin/python wikiclick.py million-status
.venv/bin/python -m pytest -q
```

Each model/navigation command uses saved data and prompts for source and destination titles when needed. The next-link demo supports the saved 1,000-article dataset. `million-status` inspects the completed larger experiment without starting another run. The launcher resolves paths from this project directory even if invoked elsewhere by absolute path.

Collection and training are explicit commands:

```bash
.venv/bin/python wikiclick.py api --help
.venv/bin/python wikiclick.py collect --help
.venv/bin/python wikiclick.py raw-train --help
.venv/bin/python wikiclick.py next-link-train --help
.venv/bin/python wikiclick.py evolve --help
```

Existing `python -m codes....` commands in the detailed guides also work from the project root. Direct execution of files inside `codes/` is replaced by the launcher or `python -m`.

## Results and figures

See the [comparison overview](reviews/README.md), [figure guide](figures/README.md), and [saved-model index](models/README.md).

Training curves show measured loss over epochs. Route-quality charts show whether navigation reaches the goal, finds a shortest route, and saves time/work. A visually smoother loss curve does not establish better navigation. The evolutionary pilot did not establish an improvement over the existing ranker.

Distances and shortest-path claims apply to each saved graph, not all Wikipedia. Different graph sizes and query sets are separate experiments. The million-article workflow is complete; this cleanup did not retrain models or change measured outcomes.

## Cleanup record

The [organization record](reviews/organization/README.md) explains the moves, removed duplicates, preserved history, and compatibility changes. Unique older work is archived rather than guessed to be disposable. Raw Wikipedia sources remain available for reproducibility.
