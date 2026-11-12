import json
from types import SimpleNamespace

import networkx as nx
import pytest

import codes.api.api_endpoint as api_endpoint
from codes.search.graph_demo import build_graph
from codes.api.sample_dataset import collect_sample, export_training, freeze_graph, write_outputs
from codes.api.validate_dataset import validate


def article(node, links):
    targets = [{"pageid": target, "title": f"Article {target}", "url": f"https://example.org/{target}"}
               for target in links]
    return {"pageid": node, "title": f"Article {node}", "requested_title": f"Article {node}",
            "url": f"https://example.org/{node}", "links": [p["title"] for p in targets],
            "link_targets": targets, "links_complete": True, "collected_at": "2026-11-05T00:00:00+00:00"}


def test_resolved_links_follow_continuation_and_exclude_missing(monkeypatch):
    source = SimpleNamespace(exists=lambda: True, ns=0, title="Canonical source", pageid=1, fullurl="https://example.org/1")
    client = SimpleNamespace(page=lambda title: source)
    batches = iter([
        {"continue": {"gplcontinue": "token", "continue": "gplcontinue||"},
         "query": {"pages": [{"pageid": 2, "title": "Canonical target", "ns": 0, "fullurl": "https://example.org/2"},
                               {"title": "Missing", "ns": 0, "missing": True}]}},
        {"query": {"pages": [{"pageid": 2, "title": "Canonical target", "ns": 0, "fullurl": "https://example.org/2"},
                               {"pageid": 3, "title": "Talk:X", "ns": 1, "fullurl": "https://example.org/3"}]}},
    ])
    calls = []
    def query(http, params):
        calls.append(dict(params))
        return next(batches)
    monkeypatch.setattr(api_endpoint, "query_api", query)
    record = api_endpoint.get_resolved_page_links(client, None, "Alias")
    assert record["title"] == "Canonical source"
    assert [p["pageid"] for p in record["link_targets"]] == [2]
    assert record["missing_link_titles"] == ["Missing"]
    assert calls[1]["gplcontinue"] == "token"
    assert calls[0]["redirects"] == 1


def test_collection_resumes_and_stops_at_exact_size(tmp_path):
    calls = []
    def fetch(title):
        calls.append(title)
        node = int(title.split()[-1])
        return article(node, [node + 1])
    data = collect_sample(fetch, tmp_path, count=3, seeds=["Article 1"])
    assert [p["pageid"] for p in data["pages"]] == [1, 2, 3]
    def unexpected(title):
        pytest.fail("A completed cache should avoid network access")
    resumed = collect_sample(unexpected, tmp_path, count=3, seeds=["Article 1"])
    assert resumed["pages"] == data["pages"]
    assert len(calls) == 3


def test_canonical_ids_create_edges_despite_different_link_spelling():
    a, b = article(1, [2, 2, 1, 99]), article(2, [])
    a["links"] = ["Alias of Article 2"]
    graph, _ = build_graph({"pages": [a, b]})
    assert set(graph.edges) == {(1, 2)}
    assert graph[1][2]["weight"] == 1


def test_labels_match_exact_distances_and_destination_splits(tmp_path):
    pages = [article(1, [2, 3]), article(2, [4]), article(3, [5]),
             article(4, []), article(5, [4])] + [article(i, []) for i in range(6, 11)]
    dataset = {"pages": pages}
    graph, frozen = freeze_graph(dataset)
    manifest = export_training(graph, frozen["graph_id"], tmp_path)
    sets = [set(info["destination_ids"]) for info in manifest["splits"].values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    observed = {}
    for split in ("train", "validation", "test"):
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            a, b, d = (row[key]["pageid"] for key in ("current_article", "candidate_next_article", "destination"))
            assert graph.has_edge(a, b) and a != d
            assert d in manifest["splits"][split]["destination_ids"]
            expected = nx.shortest_path_length(graph, b, d) if nx.has_path(graph, b, d) else None
            assert row["label_distance"] == expected
            assert row["reachable_in_graph"] == (expected is not None)
            observed[(a, b, d)] = row["label_distance"]
    assert observed[(1, 2, 4)] == 1
    assert observed[(1, 3, 4)] == 2
    assert observed[(1, 2, 6)] is None
    assert observed[(1, 2, 2)] == 0


def test_saved_dataset_validator_detects_corrupted_labels(tmp_path):
    pages = [article(i, [i+1] if i < 10 else []) for i in range(1, 11)]
    write_outputs({"pages": pages}, tmp_path, rows_per_destination=5)
    report = validate(tmp_path)
    assert report["articles"] == 10 and report["all_labels_match_networkx"]
    path = tmp_path / "train.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["label_distance"] = 999
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError, match="Incorrect shortest-distance"):
        validate(tmp_path)
