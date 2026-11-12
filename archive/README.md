# Archive

- `previous_outer/`: differing older versions from the outer project. These were preserved because unique work cannot safely be classified as unused by its filename alone. The active implementation is under `codes/`, and current reviews are under `reviews/`.
- `repository_history/inner.git/`: the complete Git metadata from the former inner repository, retained as historical evidence. The outer `.git/` remains the active repository.

Byte-identical duplicates, disposable caches, finished empty locks, and the redundant outer Python environment were removed. This archive is not part of the test or runtime import path.
