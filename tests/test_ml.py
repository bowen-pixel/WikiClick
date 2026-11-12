import json

import networkx as nx
import pytest

torch = pytest.importorskip("torch")

from codes.search.compare_search import choose_queries, search
from codes.models.raw.ml_model import DistanceModel, pair_features, title_features
from codes.api.sample_dataset import export_training
from codes.models.raw.train_model import load_pairs


def test_unseen_title_features_and_model_gradients():
    candidate = title_features("An unfamiliar article", 16)
    goal = title_features("Graph theory", 16)
    assert torch.equal(candidate, title_features("An unfamiliar article", 16))
    assert candidate.shape == (16,)
    model = DistanceModel(16, 8)
    predictions = model(pair_features(candidate, goal).unsqueeze(0))
    assert predictions.shape == (1,) and torch.isfinite(predictions).all()
    loss = (predictions - torch.tensor([3.0])).square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_bfs_matches_reference_and_neural_style_search_keeps_valid_routes():
    graph = nx.DiGraph([(1, 2), (2, 4), (1, 3), (3, 5), (5, 6), (6, 4), (5, 1)])
    result = search(graph, 1, 4, 20)
    assert result["clicks"] == nx.shortest_path_length(graph, 1, 4) == 2
    priority = {1: 0, 2: 10, 3: 1, 5: 1, 6: 1, 4: 0}
    guided = search(graph, 1, 4, 20, lambda nodes, goal: [priority[n] for n in nodes])
    assert guided["clicks"] == 4  # Learned ordering need not return an optimal path.
    assert all(graph.has_edge(a, b) for a, b in zip(guided["path"], guided["path"][1:]))
    assert search(graph, 1, 4, 1)["path"] is None
    assert search(graph, 1, 1, 1)["clicks"] == 0


def test_row_loader_deduplicates_and_skips_nulls(tmp_path):
    path = tmp_path / "train.jsonl"
    row = {"graph_id": "g", "split": "train", "destination": {"pageid": 4},
           "candidate_next_article": {"pageid": 2}, "label_distance": 1}
    path.write_text("\n".join(json.dumps(value) for value in [row, row, row | {"label_distance": None}]))
    pairs, stats = load_pairs(path, "g", {4}, 10, 207)
    assert pairs == [((2, 4), 1)] and stats["null_rows_skipped"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        load_pairs(path, "other-graph", {4}, 10, 207)


def test_capped_labels_remain_exact_and_query_goals_are_held_out(tmp_path):
    graph = nx.DiGraph((a, b) for a in range(10) for b in range(10) if a != b)
    nx.set_node_attributes(graph, {node: f"Article {node}" for node in graph}, "title")
    manifest = export_training(graph, "g", tmp_path, rows_per_destination=5)
    for split, info in manifest["splits"].items():
        assert info["rows"] == 5 * len(info["destination_ids"])
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["label_distance"] == nx.shortest_path_length(graph, row["candidate_next_article"]["pageid"], row["destination"]["pageid"])
    chain = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    queries = choose_queries(chain, {4}, 5, 207)
    assert all(q["target"] == 4 and q["optimal_clicks"] >= 2 for q in queries)
