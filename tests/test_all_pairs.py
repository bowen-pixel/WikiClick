import csv
import io

import networkx as nx
import pytest

from analyze_all_pairs import analyze


def test_directed_distances_exclude_self_and_unreachable_pairs():
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    graph.add_node(4)
    stream = io.StringIO()
    report, sources = analyze(graph, csv.writer(stream))
    assert report["total_ordered_pairs"] == 12
    assert report["reachable_pairs"] == report["unreachable_pairs"] == 6
    assert report["mean_optimal_clicks_reachable"] == 1.5
    assert report["median_optimal_clicks_reachable"] == 1.5
    assert report["distance_histogram"] == {1: 3, 2: 3}
    assert sources[-1]["reachable_targets"] == 0
    assert sources[-1]["mean_optimal_clicks_reachable"] is None
    rows = list(csv.reader(io.StringIO(stream.getvalue())))
    assert len(rows) == 12 and all(a != b for a, b, distance in rows)
    assert sum(distance == "" for a, b, distance in rows) == 6


def test_unreachable_graph_has_no_finite_average():
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2])
    report, _ = analyze(graph)
    assert report["total_ordered_pairs"] == 2
    assert report["mean_optimal_clicks_reachable"] is None
    assert report["maximum_finite_distance"] is None


def test_asymmetry_and_nonunit_cost_rejection():
    graph = nx.DiGraph([(1, 2), (2, 3)])
    report, _ = analyze(graph)
    assert report["reachable_pairs"] == 3
    assert report["mean_optimal_clicks_reachable"] == pytest.approx(4/3)
    graph[1][2]["weight"] = 2
    with pytest.raises(ValueError, match="unit-weight"):
        analyze(graph)
