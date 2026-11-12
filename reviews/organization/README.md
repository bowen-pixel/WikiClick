# Organization and cleanup

The former `WikiClick/WikiClick/` working files have been consolidated into the outer project. The inner copy contained the more recent implementation and experiment results. The outer Git repository remains active, and the inner Git metadata is preserved under `archive/repository_history/inner.git/`.

## Organization

- Collection and API code: `codes/api/`.
- BFS, article input, and search comparisons: `codes/search/`.
- PyTorch implementations: `codes/models/raw/`, `next_link/`, and `evolution/`.
- Saved weights and associated experiment records: `models/`.
- Graphs, JSON/JSONL labels, source records, caches, and sparse data: `data/`.
- Human-readable reviews: `reviews/comparisons/`; workflow guides: `reviews/guides/`.
- Standalone machine-readable comparisons: `reviews/results/`.
- Charts: `figures/raw/`, `next_link/`, and `evolution/`.
- Runtime logs: `logs/`; dependencies: `requirements/`.

Imports use the `codes` package. Workflow subprocesses use `python -m` so moved files retain their imports. Workflow working directories resolve to the project root. The `wikiclick.py` launcher offers short commands and resolves dataset paths from the root even when invoked elsewhere. Plotters write charts into `figures/` and generated reviews into `reviews/comparisons/`.

## Deletion policy and evidence

Only verified byte-identical duplicates or regeneratable material were discarded. Differing older files were moved to `archive/previous_outer/`, not overwritten. The newer environment contained all packages in the old environment; package inventories are retained in `previous_environments.json`. Virtual-environment launcher and activation paths were updated after relocation.

The cleanup removed 1,143 byte-identical duplicate files (254,227,011 bytes), plus the redundant approximately 676 MiB outer environment, generated caches, and stale empty locks. It archived 17 differing older files. Raw Wikipedia dumps, unique graph data, checkpoints, measurements, and both repositories' history were retained. No model was retrained, and no outcomes were changed; recorded path strings were updated where needed.

The million-scale workflow was already complete. No collection or training job was launched or rescheduled. “Reschedule” was interpreted as reorganizing the requested files and workflow layout.

## Maps

- `file_moves.json`: move, duplicate-removal, archive, and metadata-update records.
- `module_map.json`: previous script name to current Python module.
- `path_map.json`: previous output location to current folder.
- `previous_environments.json`: installed packages in both old environments.

Numeric outcomes remain in each experiment's original result bundle. Historical log text and archived old source files may retain previous paths; use the active README and module/path maps for current commands.

## Verification

All 55 tests passed after relocation. Regression coverage includes BFS launched from a different directory, moved workflow stages launched as modules, and pause matching for module-based process commands. The real saved evolved model completed a two-click query after relocation. All active Python files parse, and local document links resolve. Raw plot generation was exercised in a temporary directory; next-link, million-scale, and evolutionary plotters rebuilt their canonical figure directories without retraining.
