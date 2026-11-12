import gzip
import json

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
from scipy import sparse

from codes.search.benchmark_fresh_scale import search_csr
from codes.api.import_fresh_dump import SNAPSHOT, blocks, build_sparse, lookup, map_targets, numeric_rows, resolve_alias, sample_pages, unescape


def test_numeric_parser_and_mysql_titles():
    values = numeric_rows(b"\nINSERT INTO `pagelinks` VALUES\n(10,0,300),\n(20,1,400);\n-- end\n")
    assert values.tolist() == [[10, 0, 300], [20, 1, 400]]
    assert unescape(rb"Dijkstra\'s_algorithm") == "Dijkstra's_algorithm"
    assert unescape(rb"A\\B") == "A\\B"
    assert resolve_alias("Alias", {"Canonical": 0}, {"Alias": "Second", "Second": "Canonical"}, {}) == 0
    assert resolve_alias("A", {}, {"A": "B", "B": "A"}, {}) is None


def test_lookup_and_sparse_search():
    positions, valid = lookup(np.array([10, 20]), np.array([1, 10, 20, 100]))
    assert valid.tolist() == [False, True, True, False]
    graph = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8))
    result = search_csr(graph, 0, 2, 10, 10)
    assert result["path"] == [0, 1, 2] and result["expanded"] == 2
    assert search_csr(graph, 0, 2, 1, 10)["status"] == "expansion_budget"
    assert search_csr(graph, 2, 0, 10, 10)["path"] is None


def test_multiple_insert_statements_do_not_merge_ids():
    block = (b"(10,0,300);\nINSERT INTO `pagelinks` VALUES\n"
             b"(20,0,400),\n(21,0,401);\n"
             b"-- comment\nINSERT INTO `pagelinks` VALUES\n(30,0,500);\n")
    assert numeric_rows(block).tolist() == [[10, 0, 300], [20, 0, 400], [21, 0, 401], [30, 0, 500]]


def test_wrong_numeric_column_count_is_rejected():
    with pytest.raises(ValueError, match="Malformed pagelinks"):
        numeric_rows(b"(10,0,300,400);\n")


def test_fresh_import_preserves_originals_resolves_aliases_and_deduplicates(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    output = tmp_path / "output"; output.mkdir()
    payloads = {
        "page": "(10,0,'Original',0,0),\n(20,0,'New',0,0),\n(30,0,'Alias',1,0);\n",
        "redirect": "(30,0,'New','','');\n",
        "linktarget": "(100,0,'Alias'),\n(101,0,'New'),\n(102,0,'Original');\n",
        "pagelinks": "(10,0,100),\n(10,0,101),\n(20,0,102),\n(20,0,101);\n",
    }
    for table, payload in payloads.items():
        with gzip.open(raw / f"enwiki-{SNAPSHOT}-{table}.sql.gz", "wt") as stream:
            stream.write(f"INSERT INTO `{table}` VALUES\n"+payload)
    settings = sample_pages(raw, output, {"nodes": [{"pageid": 10}], "graph_id": "old"}, 1, 207)
    assert settings["retained"] == 1 and settings["nodes"] == 2
    map_targets(raw, output)
    graph = build_sparse(raw, output)
    assert graph.toarray().tolist() == [[0, 1], [1, 0]]
    assert list(blocks(raw / f"enwiki-{SNAPSHOT}-page.sql.gz", "page", 7))
