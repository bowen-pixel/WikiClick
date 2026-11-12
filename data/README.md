# Data

- `graphs/`: saved graph JSON and metadata, training/validation/test JSONL or NPZ, exact labels, cached semantic features and retained source records. Dataset names identify scale and experiment.
- `cache/`: cached canonical API articles and prepared feature arrays.
- `raw/`: downloaded Wikimedia source tables and article-text shards needed to reproduce collection/import.

These are local, generated datasets excluded from ordinary Git commits. Labels describe shortest distances in the associated saved graph. Do not mix graph IDs or held-out destination partitions.

JSON, JSONL, compressed JSONL, NumPy, and sparse files stay together by dataset. Consolidation retained unique data and removed only byte-identical outer-folder duplicates. Raw sources were preserved; this cleanup did not recollect Wikipedia.
