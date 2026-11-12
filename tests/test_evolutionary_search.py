import copy
import random

import networkx as nx
import pytest
import torch

from codes.models.evolution.evolutionary_search import SPACE, Refinement, fit, key, load_selected, mutate, query_sample
from codes.models.raw.ml_model import DistanceModel
from codes.models.next_link.next_link_experiment import Ranker, decisions


def test_mutation_changes_one_gene_and_never_repeats():
    parent = SPACE[0]
    seen = {key(parent)}
    for _ in range(5):
        child = mutate(parent, random.Random(len(seen)), seen)
        assert sum(child[k] != parent[k] for k in parent) == 1
        assert key(child) not in seen
        seen.add(key(child))
    with pytest.raises(ValueError, match="exhausted"):
        mutate(parent, random.Random(1), {key(c) for c in SPACE})


def test_new_queries_are_reachable_distinct_and_exclude_old_pairs():
    graph = nx.DiGraph([(i, (i + 1) % 6) for i in range(6)])
    old = query_sample(graph, [0, 1, 2], 3, 4)
    new = query_sample(graph, [0, 1, 2], 3, 4, [(r["source"], r["target"]) for r in old])
    assert len({r["target"] for r in new}) == 3
    assert not {(r["source"], r["target"]) for r in old} & {(r["source"], r["target"]) for r in new}
    for r in new:
        assert nx.shortest_path_length(graph, r["source"], r["target"]) == r["optimal_clicks"] >= 2


def test_adapter_starts_at_baseline_and_training_freezes_backbone():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    base = Ranker(DistanceModel().state_dict(), semantic=True)
    before = copy.deepcopy(base.state_dict())
    title, semantic = torch.randn(6, 128), torch.randn(6, 128)
    c, d = torch.tensor([0, 1]), torch.tensor([4, 4])
    for config in SPACE:
        model = Refinement(base, config)
        assert torch.equal(model(c, d, title, semantic), base(c, d, title, semantic))
    graph = nx.DiGraph([(i, (i + 1) % 6) for i in range(6)] + [(0, 2), (1, 3), (2, 4)])
    index = {i: i for i in graph}
    train, _ = decisions(graph, [0, 1], index, 3, 1)
    val, _ = decisions(graph, [2, 3], index, 3, 2)
    queries = query_sample(graph, [2, 3], 2, 3)
    model, record = fit(base, SPACE[0], train, val, queries, graph, index, title, semantic, 2, 8)
    assert len(record["history"]) == 3
    assert record["selected_epoch"] in [0, 1, 2]
    assert all(torch.equal(before[k], v) for k, v in model.base.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    assert all(not p.requires_grad for p in model.base.parameters())


def test_saved_adapter_and_baseline_rollback_reload(tmp_path):
    base = Ranker(DistanceModel().state_dict(), semantic=True)
    for config in (None, SPACE[-1]):
        model = Refinement(base, config) if config else base
        path = tmp_path / "selected.pt"
        torch.save({"state_dict": model.state_dict(), "config": config, "graph_id": "fixture"}, path)
        loaded = load_selected(path, "fixture")
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
        with pytest.raises(ValueError, match="graph differ"):
            load_selected(path, "other")
