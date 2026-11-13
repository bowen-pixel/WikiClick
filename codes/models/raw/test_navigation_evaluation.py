"""Focused evaluation tests kept within the user-authorized raw code directory."""
import copy

import networkx as nx
import pytest
import torch

from codes.models.raw.navigation_test import (ENCODER_NAME, SEMANTIC_FEATURE_VERSION,
    evaluate_bfs, evaluate_model, load_saved_model, paired_route_gain, state_hash)
from codes.models.raw.regularization_experiment import RegularizedDistanceModel


class FirstFeature(torch.nn.Module):
    def forward(self, x):
        return x[:, 0]


def test_unreachable_link_and_nonoptimal_search_are_reported_honestly():
    graph = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (5, 4)])
    scores = {0: 0, 1: 10, 2: 0, 3: -1, 4: 0, 5: 0}
    vectors = {n: torch.tensor([float(scores[n])]) for n in graph}
    queries = [{'source': 0, 'target': 4, 'optimal_clicks': 2}]
    result = evaluate_model(FirstFeature(), queries, [1, 25], graph, vectors,
                            (torch.zeros(1, 4), torch.zeros(1)))
    assert result['optimal_next_link_accuracy'] == 0
    assert result['unreachable_next_link_selections'] == 1
    assert result['mean_next_link_regret_reachable_selections_only'] is None
    assert result['by_budget']['1']['success_rate'] == 0
    assert result['by_budget']['1']['mean_extra_clicks_successes_only'] is None
    assert result['by_budget']['25']['success_rate'] == 1
    assert result['by_budget']['25']['optimal_route_rate_all_queries'] == 0
    assert result['by_budget']['25']['routes'][0]['clicks'] == 3
    bfs = evaluate_bfs(graph, queries, [1, 25])
    assert bfs['1']['success_rate'] == 0
    assert bfs['25']['optimal_route_rate_all_queries'] == 1


def test_all_equally_short_links_are_valid_labels():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    vectors = {n: torch.tensor([float(n)]) for n in graph}
    result = evaluate_model(FirstFeature(), [{'source': 0, 'target': 3, 'optimal_clicks': 2}],
                            [25], graph, vectors, (torch.zeros(1, 4), torch.zeros(1)))
    assert result['optimal_next_link_accuracy'] == 1
    assert result['by_budget']['25']['optimal_route_rate_all_queries'] == 1


def test_loader_checks_actual_R1_configuration_and_disables_dropout(tmp_path):
    model = RegularizedDistanceModel(2, 4, .1, 0)
    checkpoint = dict(state_dict=model.state_dict(), graph_id='fixture', dimensions=2, hidden=4,
                      model_seed=207, data_seed=611, feature_method='semantic',
                      feature_version=SEMANTIC_FEATURE_VERSION, encoder_name=ENCODER_NAME,
                      dropout1=.1, dropout2=0., weight_decay=0.)
    path = tmp_path / 'best.pt'
    torch.save(checkpoint, path)
    loaded = load_saved_model(path, 'fixture', 2, 4, 207, 'R1_dropout')
    before = state_hash(loaded)
    x = torch.randn(10, 8)
    assert torch.equal(loaded(x), loaded(x))
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    assert state_hash(loaded) == before
    checkpoint['dropout2'] = .1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='dropout2'):
        load_saved_model(path, 'fixture', 2, 4, 207, 'R1_dropout')


def test_identical_paired_routes_have_zero_gain():
    rows = [{'source': 0, 'target': 2, 'found': True, 'clicks': 2, 'optimal_clicks': 2},
            {'source': 1, 'target': 3, 'found': False, 'clicks': None, 'optimal_clicks': 2}]
    result = {'207': {m: {'by_budget': {'25': {'routes': copy.deepcopy(rows)}}}
                      for m in ['R0_baseline', 'R7_strong_dropout']}}
    paired = paired_route_gain(result, 'R0_baseline', 'R7_strong_dropout', 25, [207])
    assert paired['success']['conditional_95_percent_interval'] == [0., 0.]
    assert paired['optimal_route']['gain_percentage_points'] == 0
    result['207']['R7_strong_dropout']['by_budget']['25']['routes'].reverse()
    with pytest.raises(ValueError, match='Unmatched'):
        paired_route_gain(result, 'R0_baseline', 'R7_strong_dropout', 25, [207])
