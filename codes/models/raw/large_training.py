"""Build supervised sparse-graph labels, cache title features, and fine-tune."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from codes.models.raw.ml_model import DistanceModel, FEATURE_VERSION
from codes.api.sample_dataset import save_json
from codes.models.raw.train_model import mean_absolute_error


def fast_title_features(title, dimensions=128):
    """Equivalent fixed hashing; count in NumPy rather than many tensor updates."""
    text = " "+" ".join(title.lower().split())+" "
    tokens = re.findall(r"\w+", text)
    tokens += [text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1))]
    indices, signs = [], []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        indices.append(int.from_bytes(digest[:4], "little") % dimensions)
        signs.append(1 if digest[4] % 2 else -1)
    counts = np.bincount(indices, weights=signs, minlength=dimensions).astype(np.float32)
    result = torch.from_numpy(counts)
    return (result/result.norm().clamp_min(1)).numpy()


def atomic_npz(path, **values):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(path)


def generate_labels(data, retained, output, counts=(200, 25, 25), pairs_per_goal=2048, seed=208, queries=100):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data/"metadata.json").read_text())
    retained_metadata = json.loads((retained/"metadata.json").read_text())
    ids = np.load(data/"page_ids.npy")
    old_ids = np.load(retained/"page_ids.npy")
    if not np.isin(old_ids, ids, assume_unique=True).all():
        raise ValueError("Expanded graph did not retain every previous article")
    eligible = np.flatnonzero(~np.isin(ids, old_ids, assume_unique=True))
    config = {"graph_id": metadata["graph_id"], "retained_graph_id": retained_metadata["graph_id"],
              "seed": seed, "counts": list(counts), "pairs_per_goal": pairs_per_goal,
              "new_articles": len(eligible), "label": "Exact candidate-to-destination distance on expanded graph",
              "destination_pool": "Only the newly added articles; disjoint destination splits"}
    config_path = output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Label settings changed; use a new output directory")
    save_json(config_path, config)
    graph = sparse.load_npz(data/"graph.csr.npz").tocsr()
    reverse = graph.transpose().tocsr()
    rng = np.random.default_rng(seed)
    candidates = rng.choice(eligible, size=min(len(eligible), sum(counts)*4), replace=False)
    goal_cache = output/"goals"; goal_cache.mkdir(exist_ok=True)
    accepted = []
    skipped = []
    for goal in candidates:
        cache = goal_cache/f"{int(ids[goal])}.npz"
        if cache.exists():
            saved = np.load(cache)
            pair_array, labels = saved["pairs"], saved["labels"]
            if str(saved["graph_id"]) != metadata["graph_id"]:
                raise ValueError("Cached goal labels belong to a different graph")
        else:
            distances = shortest_path(reverse, directed=True, indices=int(goal), unweighted=True)
            reachable = np.flatnonzero(np.isfinite(distances))
            if np.count_nonzero(distances[reachable] >= 2) < 4:
                skipped.append(int(ids[goal]))
                continue
            sample_rng = np.random.default_rng(seed+int(ids[goal]))
            sample = sample_rng.choice(reachable, size=min(pairs_per_goal, len(reachable)), replace=False)
            # Include the zero-distance identity once for calibration.
            if goal not in sample:
                sample[0] = goal
            pair_array = np.column_stack((sample, np.full(len(sample), goal))).astype(np.int32)
            labels = distances[sample].astype(np.float32)
            atomic_npz(cache, pairs=pair_array, labels=labels, graph_id=np.array(metadata["graph_id"]),
                       reachable_vertices=np.array(len(reachable)))
        accepted.append((int(goal), pair_array, labels))
        print(f"Exact supervised goal {len(accepted)}/{sum(counts)}: {int(ids[goal])}", flush=True)
        save_json(output/"progress.json", {"accepted_goals": len(accepted), "required_goals": sum(counts),
                                           "latest_page_id": int(ids[goal]), "graph_id": metadata["graph_id"]})
        if len(accepted) == sum(counts):
            break
    if len(accepted) != sum(counts):
        raise ValueError("Not enough reachable destination goals; cached work preserved")
    partitions = {}
    cursor = 0
    query_candidates = []
    for split, count in zip(("train", "validation", "test"), counts):
        selected = accepted[cursor:cursor+count]; cursor += count
        pairs = np.concatenate([value[1] for value in selected])
        labels = np.concatenate([value[2] for value in selected])
        goals = [int(ids[value[0]]) for value in selected]
        atomic_npz(output/f"{split}.npz", pairs=pairs, labels=labels, graph_id=np.array(metadata["graph_id"]))
        partitions[split] = {"destination_ids": goals, "examples": len(labels)}
        if split == "test":
            query_candidates = [(int(a), int(d), int(label)) for (a, d), label in zip(pairs, labels) if label >= 2]
    if len(query_candidates) < queries:
        raise ValueError("Not enough test pairs for the requested route benchmark")
    query_indices = np.random.default_rng(seed+1).choice(len(query_candidates), size=queries, replace=False)
    query_rows = []
    for index in query_indices:
        source, target, distance = query_candidates[index]
        query_rows.append({"source": source, "target": target, "optimal_clicks": distance,
                           "source_page_id": int(ids[source]), "target_page_id": int(ids[target])})
    save_json(output/"evaluation_queries.json", {"graph_id": metadata["graph_id"], "queries": query_rows})
    manifest = {**config, "splits": partitions, "skipped_destination_ids": skipped,
                "null_policy": "Unreachable vertices excluded from numeric labels, never relabeled zero",
                "evaluation": "Shared graph, held-out new destination articles; not all pairs or every article used as a goal"}
    save_json(output/"manifest.json", manifest)
    return manifest


def prepare_features(data, labels_dir, output, dimensions=128):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data/"metadata.json").read_text())
    node_count = metadata["nodes"]
    needed = np.zeros(node_count, dtype=bool)
    for split in ("train", "validation", "test"):
        # Only pair identities are used here; no test labels or fitted vocabulary.
        pairs = np.load(labels_dir/f"{split}.npz")["pairs"]
        needed[pairs.ravel()] = True
    config = {"graph_id": metadata["graph_id"], "dimensions": dimensions, "feature_version": FEATURE_VERSION}
    config_path = output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Feature cache settings differ")
    save_json(config_path, config)
    path = output/"vectors.npy"; filled_path = output/"filled.npy"
    vectors = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(node_count, dimensions))
    filled = np.lib.format.open_memmap(filled_path, mode="r+" if filled_path.exists() else "w+", dtype=bool, shape=(node_count,))
    completed = 0
    with gzip.open(data/"nodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if needed[index] and not filled[index]:
                vectors[index] = fast_title_features(json.loads(line)["title"], dimensions)
                filled[index] = True
                completed += 1
                if completed % 10000 == 0:
                    vectors.flush(); filled.flush()
                    print(f"Computed {completed:,} additional title vectors", flush=True)
    vectors.flush(); filled.flush()
    if not np.all(filled[needed]):
        raise ValueError("Feature cache is incomplete")
    save_json(output/"manifest.json", {**config, "needed_vectors": int(needed.sum()),
                                       "scope": "Fixed title hashing only; no model fitting or test-label use"})


def pair_tensors(path, vectors):
    saved = np.load(path)
    pairs, labels = saved["pairs"], saved["labels"]
    x = np.empty((len(pairs), vectors.shape[1]*4), dtype=np.float32)
    for start in range(0, len(pairs), 4096):
        batch = pairs[start:start+4096]
        candidate = vectors[batch[:, 0]]
        destination = vectors[batch[:, 1]]
        x[start:start+len(batch)] = np.concatenate((candidate, destination, np.abs(candidate-destination), candidate*destination), axis=1)
    return torch.from_numpy(x), torch.from_numpy(labels.copy())


def fine_tune(data, labels_dir, feature_dir, output, initial, epochs=25, seed=208):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest = json.loads((labels_dir/"manifest.json").read_text())
    metadata = json.loads((data/"metadata.json").read_text())
    if manifest["graph_id"] != metadata["graph_id"]:
        raise ValueError("Labels and graph differ")
    goal_sets = [set(value["destination_ids"]) for value in manifest["splits"].values()]
    if any(a & b for i, a in enumerate(goal_sets) for b in goal_sets[i+1:]):
        raise ValueError("Destination split leakage")
    parent = torch.load(initial, map_location="cpu", weights_only=True)
    feature_config = json.loads((feature_dir/"config.json").read_text())
    if (parent["feature_version"] != FEATURE_VERSION or feature_config["graph_id"] != metadata["graph_id"]
            or feature_config["dimensions"] != parent["dimensions"]):
        raise ValueError("Checkpoint, feature cache, and graph are incompatible")
    model = DistanceModel(parent["dimensions"], parent["hidden"])
    model.load_state_dict(parent["state_dict"])
    initial_state = copy.deepcopy(model.state_dict())
    vectors = np.load(feature_dir/"vectors.npy", mmap_mode="r")
    x_train, y_train = pair_tensors(labels_dir/"train.npz", vectors)
    x_val, y_val = pair_tensors(labels_dir/"validation.npz", vectors)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_function = nn.SmoothL1Loss()
    model.eval()
    initial_error = mean_absolute_error(model, x_val, y_val)
    best_error, best_epoch, best = initial_error, 0, copy.deepcopy(initial_state)
    history = []; updates = 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train(); total = 0.
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward(); optimizer.step()
            total += loss.item()*len(y); updates += 1
        model.eval()
        error = mean_absolute_error(model, x_val, y_val)
        history.append({"epoch": epoch, "train_loss": total/len(y_train), "validation_mae": error})
        if error < best_error:
            best_error, best_epoch, best = error, epoch, copy.deepcopy(model.state_dict())
        save_json(output/"progress.json", {"epoch": epoch, "epochs": epochs, "optimizer_updates": updates,
                                           "train_loss": total/len(y_train), "validation_mae": error,
                                           "best_epoch": best_epoch, "model_weights_updated": True})
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "graph_id": metadata["graph_id"]}, output/"last.pt")
        print(f"Fine-tuning epoch {epoch}/{epochs}: loss={total/len(y_train):.4f}, validation MAE={error:.4f}", flush=True)
    change = sum((best[key]-initial_state[key]).square().sum().item() for key in best)**.5
    checkpoint = {"state_dict": best, "dimensions": parent["dimensions"], "hidden": parent["hidden"],
                  "feature_version": FEATURE_VERSION, "graph_id": metadata["graph_id"],
                  "training_graph_node_ids": np.load(data/"page_ids.npy").tolist(),
                  "destination_splits": {key: value["destination_ids"] for key, value in manifest["splits"].items()},
                  "seed": seed, "selected_epoch": best_epoch, "graph_nodes": metadata["nodes"],
                  "parent_checkpoint_graph_id": parent["graph_id"], "optimizer_updates": updates,
                  "selected_parameter_change_l2": change}
    torch.save(checkpoint, output/"best.pt")
    model.load_state_dict(best); model.eval()
    # Test labels are first loaded after checkpoint selection, once for final scoring.
    x_test, y_test = pair_tensors(labels_dir/"test.npz", vectors)
    test_error = mean_absolute_error(model, x_test, y_test)
    constant_error = (y_test-y_train.mean()).abs().mean().item()
    metrics = {"graph_id": metadata["graph_id"], "graph_nodes": metadata["nodes"], "torch_version": str(torch.__version__),
               "initial_validation_mae": initial_error, "validation_mae": best_error, "test_mae": test_error,
               "constant_test_mae": constant_error, "selected_epoch": best_epoch, "epochs": epochs,
               "optimizer_updates": updates, "selected_parameter_change_l2": change,
               "training_seconds": time.perf_counter()-started, "history": history,
               "examples": {key: value["examples"] for key, value in manifest["splits"].items()},
               "scope": "Actual supervised fine-tuning on sampled expanded-graph labels; held-out newly added destination articles"}
    save_json(output/"metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("labels", "features", "train"))
    parser.add_argument("--data", type=Path, default=Path("data/graphs/fresh_two_million"))
    parser.add_argument("--retained", type=Path, default=Path("data/graphs/fresh_million"))
    parser.add_argument("--labels", type=Path, default=Path("data/graphs/large_supervised"))
    parser.add_argument("--features", type=Path, default=Path("data/cache/large_features"))
    parser.add_argument("--output", type=Path, default=Path("models/raw/model_large_trained"))
    parser.add_argument("--initial", type=Path, default=Path("models/raw/model_1100/best.pt"))
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    if args.stage == "labels":
        generate_labels(args.data, args.retained, args.labels)
    elif args.stage == "features":
        prepare_features(args.data, args.labels, args.features)
    else:
        fine_tune(args.data, args.labels, args.features, args.output, args.initial, args.epochs)
