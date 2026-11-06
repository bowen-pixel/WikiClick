from types import SimpleNamespace

import networkx as nx
import pytest

from api_endpoint import collect_domain, get_page_links
from graph_demo import build_graph, find_path


def record(node, title, links=(), requested=None):
    return {"pageid": node, "title": title, "requested_title": requested or title,
            "url": f"https://en.wikipedia.org/wiki/{title}", "links": list(links)}


def test_shortest_route_weights_aliases_and_deduplication():
    data = {"pages": [
        record(1, "A", ["A", "B_alias", "B_alias", "C", "Outside"]),
        record(2, "B", ["D"], requested="B alias"),
        record(3, "C", ["E"]), record(4, "D"), record(5, "E", ["D"]),
    ]}
    graph, aliases = build_graph(data)
    assert find_path(graph, aliases, "a", "D") == [1, 2, 4]
    assert set(graph.edges) == {(1, 2), (1, 3), (2, 4), (3, 5), (5, 4)}
    assert all(edge["weight"] == 1 for _, _, edge in graph.edges(data=True))
    assert find_path(graph, aliases, "A", "A") == [1]
    with pytest.raises(nx.NetworkXNoPath):
        find_path(graph, aliases, "D", "A")
    with pytest.raises(ValueError, match="outside this dataset"):
        find_path(graph, aliases, "A", "Unknown")


class FakeClient:
    def __init__(self, exists=True, namespace=0):
        self.article = SimpleNamespace(exists=lambda: exists, ns=namespace,
                                       title="Canonical", pageid=42, fullurl="https://example.org/42")

    def page(self, title):
        return self.article

    def links(self, page, **kwargs):
        assert kwargs == {"plnamespace": 0, "pllimit": "max"}
        return {"Article": SimpleNamespace(ns=0), "Talk:Article": SimpleNamespace(ns=1)}


def test_api_returns_plain_data_and_filters_namespace():
    page = get_page_links(FakeClient(), "Alias")
    assert page == record(42, "Canonical", ["Article"], requested="Alias") | {"url": "https://example.org/42"}


@pytest.mark.parametrize("client,message", [
    (FakeClient(exists=False), "does not exist"),
    (FakeClient(namespace=1), "Only article"),
])
def test_invalid_articles_are_not_empty_successes(client, message):
    with pytest.raises(ValueError, match=message):
        get_page_links(client, "Invalid")


def test_collection_is_bounded_and_deduplicates_requests():
    assert len(collect_domain(FakeClient(), ["A", "A"])["pages"]) == 1
    for titles in ([], [str(i) for i in range(21)]):
        with pytest.raises(ValueError, match="between 1 and 20"):
            collect_domain(FakeClient(), titles)


def test_api_failure_propagates():
    class BrokenClient(FakeClient):
        def links(self, page, **kwargs):
            raise RuntimeError("Network unavailable")

    with pytest.raises(RuntimeError, match="Network unavailable"):
        get_page_links(BrokenClient(), "A")
