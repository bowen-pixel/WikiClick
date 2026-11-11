import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
torch = pytest.importorskip("torch")
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from large_training import fast_title_features, fine_tune, generate_labels, prepare_features
from ml_model import DistanceModel, FEATURE_VERSION, title_features


@pytest.mark.parametrize("title", ["Graph theory", "Dijkstra's algorithm", "", "Québec and 数学"])
def test_fast_hash_features_match_existing_model_inputs(title):
    for dimensions in (16, 128):
        assert np.array_equal(fast_title_features(title, dimensions), title_features(title, dimensions).numpy())


def test_new_graph_labels_training_updates_and_shared_query_evaluation(tmp_path):
    data = tmp_path/"data"; data.mkdir()
    retained = tmp_path/"retained"; retained.mkdir()
    labels = tmp_path/"labels"; features = tmp_path/"features"; model_dir = tmp_path/"model"
    ids = np.arange(100, 140)
    np.save(data/"page_ids.npy", ids)
    np.save(retained/"page_ids.npy", ids[:10])
    (retained/"metadata.json").write_text(json.dumps({"graph_id": "retained"}))
    (data/"metadata.json").write_text(json.dumps({"graph_id": "expanded", "nodes": 40, "snapshot": "20261101"}))
    matrix = sparse.csr_matrix((np.ones(40, dtype=np.uint8), (np.arange(40), (np.arange(40)+1)%40)), shape=(40, 40))
    sparse.save_npz(data/"graph.csr.npz", matrix)
    with gzip.open(data/"nodes.jsonl.gz", "wt", encoding="utf-8") as stream:
        for node in ids:
            stream.write(json.dumps({"pageid": int(node), "title": f"Article {node}"})+"\n")
    manifest = generate_labels(data, retained, labels, counts=(4, 2, 2), pairs_per_goal=10, queries=4)
    sets = [set(value["destination_ids"]) for value in manifest["splits"].values()]
    assert not any(a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert not set.union(*sets) & set(ids[:10])
    for split in ("train", "validation", "test"):
        saved = np.load(labels/f"{split}.npz")
        for (candidate, goal), distance in zip(saved["pairs"], saved["labels"]):
            assert distance == shortest_path(matrix, directed=True, indices=int(candidate), unweighted=True)[goal]
    prepare_features(data, labels, features, dimensions=16)
    initial = tmp_path/"initial.pt"
    torch.manual_seed(208)
    parent = {"state_dict": DistanceModel(16, 8).state_dict(), "dimensions": 16, "hidden": 8,
              "graph_id": "old", "feature_version": FEATURE_VERSION,
              "training_graph_node_ids": ids[:10].tolist()}
    torch.save(parent, initial)
    metrics = fine_tune(data, labels, features, model_dir, initial, epochs=2)
    assert metrics["optimizer_updates"] == 2
    assert metrics["selected_parameter_change_l2"] > 0
    assert (model_dir/"last.pt").is_file()
    script = Path(__file__).resolve().parents[1]/"benchmark_fresh_scale.py"
    reports = []
    for checkpoint, name in ((initial, "frozen"), (model_dir/"best.pt", "trained")):
        output = tmp_path/f"{name}.json"
        subprocess.run([sys.executable, str(script), "--data", str(data), "--checkpoint", str(checkpoint),
                        "--query-file", str(labels/"evaluation_queries.json"), "--queries", "4", "--budget", "40",
                        "--seconds-per-query", "5", "--output", str(output)], check=True, capture_output=True, text=True)
        reports.append(json.loads(output.read_text()))
    assert [(r["source"], r["target"]) for r in reports[0]["results"]] == [(r["source"], r["target"]) for r in reports[1]["results"]]
    assert reports[1]["summary"]["bfs"]["successes"] == 4
