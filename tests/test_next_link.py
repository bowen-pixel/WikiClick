import networkx as nx
import torch

from next_link_experiment import decisions, ranking_loss, Ranker, navigate
from ml_model import DistanceModel


def test_multiple_optimal_links_and_unreachable_negative():
    graph = nx.DiGraph([(0,1),(0,2),(0,3),(1,4),(2,4),(3,3)])
    rows, pairs = decisions(graph, [4], {i:i for i in range(5)}, 10, 5)
    row = next(r for r in rows if r["current"] == 0)
    assert row["candidates"] == [1,2,3]
    assert row["optimal"] == [True,True,False]
    assert (3,4) not in pairs
    assert row["distance"] == 2


def test_grouped_loss_rewards_optimal_links_and_is_shift_invariant():
    good = torch.tensor([2.,2.,-2.], requires_grad=True)
    bad = torch.tensor([-2.,-2.,2.])
    mask = [True,True,False]
    loss = ranking_loss(good, mask)
    assert loss < ranking_loss(bad, mask)
    assert torch.allclose(loss, ranking_loss(good+100, mask), atol=1e-5)
    loss.backward()
    assert good.grad[2] > 0


def test_semantic_ranker_starts_at_raw_scores():
    torch.manual_seed(4)
    state = DistanceModel().state_dict()
    raw, semantic = Ranker(state), Ranker(state, True)
    title, context = torch.randn(5,128), torch.randn(5,128)
    c, d = torch.tensor([0,1]), torch.tensor([4,4])
    assert torch.allclose(raw(c,d,title,context),semantic(c,d,title,context))


def test_navigation_returns_only_valid_edges():
    graph = nx.DiGraph([(0,1),(1,2)])
    model = Ranker(DistanceModel().state_dict())
    vectors = torch.zeros(3,128)
    result = navigate(graph,[{"source":0,"target":2,"optimal_clicks":2}],{i:i for i in range(3)},model,vectors,vectors)[0]
    assert result["path"] == [0,1,2]
    assert result["clicks"] == 2
