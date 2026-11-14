# Consolidated training data

`training_data.json` is the single public JSON bundle (schema version 1). It preserves recoverable saved source corpora from the original distance/next-link runs and later offline behavioral-cloning, trajectory and scaling work. It is not a claim that every available row was used by every checkpoint. Exact per-optimizer visit histories were not saved consistently and cannot be reconstructed honestly.

## Reading the JSON

- `graphs`: the full 100/1,100/1,000-article graph snapshots and original manifests; million-scale node-index mappings; later offline graph identities/mappings. Full million-node edge graphs are excluded.
- `sources`: original project-relative path, original file SHA256, encoding, split/experiment, and actual embedded data. `columns` rows preserve current/destination/candidate page IDs and distance. `numpy-json` preserves values, shape/dtype where recorded. Ranking decisions preserve all candidates and optimal masks.
- `training_evidence`: saved correction indices, training locks and checkpoint-selection identities. These document provenance; they are not optimizer histories.
- `gaps`: explicit exclusions and recovery limits. Validation/test partitions are not counted as training labels.

Original `.jsonl` rows are compacted into numeric columns to avoid repeating article titles in every row; titles/URLs are available from the small graph metadata. Ranking indices are zero-based positions in the corresponding frozen node ordering, not page IDs. Preserve these mappings. `fresh_million` and `fresh_two_million` node-index arrays must not be interchanged. Later scaling state rows refer to the 10,000-node `large_graph_8d` mapping.

Model weights, full article introductions, pretrained encoder tensors, full sparse million-node edges, temporary files and evaluation-only oracle caches are not training-record exports and remain local. Source-pool coverage does not guarantee byte-for-byte replay of every historical update. Training that needs historical feature content requires the local caches; collecting current text creates a new representation snapshot.

This file is about 86 MB. GitHub warns for files over 50 MiB and rejects files over 100 MiB; this bundle is below the hard limit. If it grows, use release assets/LFS rather than pretending an incomplete subset is complete. See [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

[Deployment/restoration commands](../Deploy.md) restore the small graph datasets without overwriting local files. Wikipedia-derived content retains its upstream attribution; the JSON records source and license-information URLs. Project code and pretrained model weights are separate from Wikipedia data.

## Embedded sources

| Original local source | Partition / role | Records or representation |
|---|---|---|
| `data/graphs/sample_100/train.jsonl` | train | 68093 |
| `data/graphs/sample_100/validation.jsonl` | validation | 8351 |
| `data/graphs/sample_100/test.jsonl` | test | 8498 |
| `data/graphs/sample_1100/train.jsonl` | train | 176000 |
| `data/graphs/sample_1100/validation.jsonl` | validation | 22000 |
| `data/graphs/sample_1100/test.jsonl` | test | 22000 |
| `data/graphs/ranking_1000/train_decisions.json` | train | 8000 |
| `data/graphs/ranking_1000/validation_decisions.json` | validation | 1000 |
| `data/graphs/ranking_1000/test_decisions.json` | test | 1000 |
| `data/graphs/large_supervised/train.npz` | train | arrays / metadata |
| `data/graphs/semantic_million/ranking_labels/train.json` | train | 8192 |
| `data/graphs/large_supervised/validation.npz` | validation | arrays / metadata |
| `data/graphs/semantic_million/ranking_labels/validation.json` | validation | 1024 |
| `data/graphs/large_supervised/test.npz` | test | arrays / metadata |
| `data/graphs/semantic_million/ranking_labels/test.json` | test | 2048 |
| `codes/models/navigation/experiments/rfbc_7b/demonstrations/B1.json` | train | 4096 |
| `codes/models/navigation/experiments/rfbc_7b/demonstrations/B2.json` | train | 4096 |
| `codes/models/navigation/experiments/rfbc_7b/demonstrations/B3.json` | train | 4096 |
| `codes/models/navigation/experiments/rfbc_7b/demonstrations/statistics.json` | train | arrays / metadata |
| `codes/models/navigation/experiments/rfbc_7b/demonstrations/walks.json` | train | 4187 |
| `codes/models/navigation/experiments/trajectory_7c/demonstrations/B2.json` | train | 4096 |
| `codes/models/navigation/experiments/trajectory_7c/demonstrations/B4.json` | train | 4096 |
| `codes/models/navigation/experiments/trajectory_7c/demonstrations/B5.json` | train | 4096 |
| `codes/models/navigation/experiments/trajectory_7c/demonstrations/auxiliary.json` | train | 4096 |
| `codes/models/navigation/experiments/trajectory_7c/demonstrations/statistics.json` | train | arrays / metadata |
| `codes/models/navigation/experiments/navigation_scaling_10a/datasets/shortest.npy` | train | 300000 |
| `codes/models/navigation/experiments/navigation_scaling_10a/datasets/walks.npy` | train | 300000 |
| `codes/models/navigation/experiments/navigation_scaling_10a/datasets/hard.npy` | train | 300000 |
| `codes/models/navigation/experiments/navigation_scaling_10a/datasets/train_distances.npy` | train | 256 |
| `data/graphs/semantic_million/ranking_labels/complete.json` | index/split provenance | arrays / metadata |
| `data/graphs/semantic_million/ranking_labels/config.json` | index/split provenance | arrays / metadata |
| `data/graphs/semantic_million/ranking_labels/progress.json` | index/split provenance | arrays / metadata |

Bundle SHA256: `bed415c7e142a8e5bba4d3866ce4f8181bb1a3418e4a1e13780c76483513d9c3`. Original-file checksums remain attached to their original encoding, not to the compacted JSON representation.
