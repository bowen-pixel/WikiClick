"""Stream fresh Wikimedia SQL dumps into a sampled, canonical sparse graph.

No SQL server is required. This importer targets the inspected 20261101 schemas.
It preserves existing nonredirect page IDs and reservoir-samples extra articles.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from scipy import sparse

from download_fresh_dump import SNAPSHOT

PAGE = re.compile(rb"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',([01]),")
REDIRECT = re.compile(rb"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',(?:NULL|'((?:[^'\\]|\\.)*)'),")
TARGET = re.compile(rb"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)'\)")


def unescape(value):
    if b"\\" in value:
        mapping = {b"0": b"\0", b"n": b"\n", b"r": b"\r", b"t": b"\t", b"Z": b"\x1a"}
        value = re.sub(rb"\\(.)", lambda match: mapping.get(match[1], match[1]), value)
    return value.decode("utf-8")


def blocks(path, table, block_size=4*1024*1024):
    """Yield complete SQL data lines; never retain an uncompressed whole dump."""
    marker = f"INSERT INTO `{table}` VALUES".encode()
    started = False
    carry = b""
    with gzip.open(path, "rb") as stream:
        while True:
            chunk = stream.read(block_size)
            if not chunk:
                if carry and started:
                    yield carry
                break
            data = carry+chunk
            if not started:
                index = data.find(marker)
                if index < 0:
                    carry = data[-len(marker):]
                    continue
                data = data[index+len(marker):]
                started = True
            split = data.rfind(b"\n")
            if split < 0:
                carry = data
                continue
            yield data[:split+1]
            carry = data[split+1:]
    if not started:
        raise ValueError(f"No INSERT data found for {table}")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    temporary.replace(path)


def sample_pages(raw, output, original, extra, seed):
    checkpoint = output / "page_stage.json"
    if checkpoint.exists():
        settings = json.loads(checkpoint.read_text())
        if settings["extra"] != extra or settings["seed"] != seed or settings["original_graph_id"] != original["graph_id"]:
            raise ValueError("Use a different output directory for different sampling settings")
        return settings
    retained_ids = (set(map(int, original["page_ids"])) if "page_ids" in original
                    else {node["pageid"] for node in original["nodes"]})
    retained = {}
    reservoir = []
    redirect_names = {}
    seen = rows = 0
    rng = random.Random(seed)
    checkpoint_rows = 0
    for block in blocks(raw / f"enwiki-{SNAPSHOT}-page.sql.gz", "page"):
        for match in PAGE.finditer(block):
            rows += 1
            if int(match[2]) != 0:
                continue
            node, title = int(match[1]), unescape(match[3])
            if match[4] == b"1":
                redirect_names[node] = title
            elif node in retained_ids:
                retained[node] = title
            else:
                seen += 1
                if len(reservoir) < extra:
                    reservoir.append((node, title))
                else:
                    index = rng.randrange(seen)
                    if index < extra:
                        reservoir[index] = (node, title)
        if rows-checkpoint_rows >= 1_000_000:
            print(f"page table: {rows:,} rows; {seen:,} eligible extra articles", flush=True)
            checkpoint_rows = rows
    missing = retained_ids-set(retained)
    if missing:
        raise ValueError(f"{len(missing)} original page IDs are missing/noncanonical in this snapshot; do not silently mix old and new graphs")
    if len(reservoir) != extra:
        raise ValueError("Not enough fresh articles for requested sample")
    nodes = sorted(list(retained.items())+reservoir)
    with gzip.open(output / "nodes.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as stream:
        for node, title in nodes:
            stream.write(json.dumps({"pageid": node, "title": title.replace("_", " ")}, ensure_ascii=False)+"\n")
    np.save(output / "page_ids.npy", np.array([node for node, title in nodes], dtype=np.int64))
    with gzip.open(output / "redirect_names.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as stream:
        for node, title in redirect_names.items():
            stream.write(json.dumps([node, title], ensure_ascii=False)+"\n")
    settings = {"seed": seed, "extra": extra, "retained": len(retained), "nodes": len(nodes),
                "eligible_extra_articles": seen, "page_rows": rows, "redirect_names": len(redirect_names),
                "original_graph_id": original["graph_id"]}
    write_json(checkpoint, settings)
    print(f"Selected {len(nodes):,} fresh articles; all {len(retained):,} originals retained", flush=True)
    return settings


def resolve_alias(title, selected, aliases, memo):
    path = []
    visiting = set()
    while title not in selected and title not in memo and title in aliases:
        if title in visiting:
            break
        visiting.add(title)
        path.append(title)
        title = aliases[title]
    result = selected.get(title, memo.get(title))
    for name in path:
        memo[name] = result
    return result


def map_targets(raw, output):
    path = output / "targets.npz"
    if path.exists():
        return
    with gzip.open(output / "nodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        selected = {json.loads(line)["title"].replace(" ", "_"): index for index, line in enumerate(stream)}
    with gzip.open(output / "redirect_names.jsonl.gz", "rt", encoding="utf-8") as stream:
        redirect_names = dict(json.loads(line) for line in stream)
    aliases = {}
    for block in blocks(raw / f"enwiki-{SNAPSHOT}-redirect.sql.gz", "redirect"):
        for match in REDIRECT.finditer(block):
            if int(match[2]) == 0 and not match[4]:
                name = redirect_names.get(int(match[1]))
                if name is not None:
                    aliases[name] = unescape(match[3])
    del redirect_names
    print(f"Loaded {len(aliases):,} local article redirects", flush=True)
    memo = {}
    ids, destinations = [], []
    for block in blocks(raw / f"enwiki-{SNAPSHOT}-linktarget.sql.gz", "linktarget"):
        for match in TARGET.finditer(block):
            if int(match[2]) != 0:
                continue
            destination = resolve_alias(unescape(match[3]), selected, aliases, memo)
            if destination is not None:
                ids.append(int(match[1]))
                destinations.append(destination)
    order = np.argsort(np.array(ids, dtype=np.int64))
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, target_ids=np.array(ids, dtype=np.int64)[order],
                 dense_ids=np.array(destinations, dtype=np.int32)[order])
    temporary.replace(path)
    print(f"Resolved {len(ids):,} hyperlink target IDs into the selected articles", flush=True)


def lookup(sorted_ids, values):
    locations = np.searchsorted(sorted_ids, values)
    valid = locations < len(sorted_ids)
    valid[valid] &= sorted_ids[locations[valid]] == values[valid]
    return locations, valid


def numeric_rows(block):
    block = re.sub(rb"INSERT INTO `pagelinks` VALUES\s*", b"", block)
    block = re.sub(rb"(?m)^(?!\().*(?:\n|$)", b"", block)
    # A semicolon separates INSERT statements just as a comma separates rows.
    # Deleting it would concatenate the last target ID with the next source ID.
    values = block.replace(b";", b",").translate(None, b"()\r\n \t").rstrip(b",")
    if not values:
        return np.empty((0, 3), dtype=np.int64)
    numbers = np.fromstring(values, dtype=np.int64, sep=",")
    if len(numbers) % 3 or len(numbers)//3 != block.count(b"("):
        raise ValueError(f"Malformed pagelinks rows: expected {block.count(b'(')} three-integer tuples, parsed {len(numbers)} integers")
    return numbers.reshape(-1, 3)


def build_sparse(raw, output):
    path = output / "graph.csr.npz"
    if path.exists():
        return sparse.load_npz(path)
    page_ids = np.load(output / "page_ids.npy")
    targets = np.load(output / "targets.npz")
    target_ids, dense_targets = targets["target_ids"], targets["dense_ids"]
    edge_file = output / "edges.uint32"
    total = kept = 0
    checkpoint = 0
    with edge_file.open("wb") as stream:
        for block in blocks(raw / f"enwiki-{SNAPSHOT}-pagelinks.sql.gz", "pagelinks"):
            rows = numeric_rows(block)
            total += len(rows)
            rows = rows[rows[:, 1] == 0]
            source_positions, valid = lookup(page_ids, rows[:, 0])
            rows, source_positions = rows[valid], source_positions[valid]
            locations, valid = lookup(target_ids, rows[:, 2])
            sources = source_positions[valid]
            destinations = dense_targets[locations[valid]]
            distinct = sources != destinations
            edges = np.column_stack((sources[distinct], destinations[distinct])).astype(np.uint32)
            stream.write(edges.tobytes())
            kept += len(edges)
            if total-checkpoint >= 10_000_000:
                print(f"pagelinks: {total:,} rows scanned; {kept:,} selected edges", flush=True)
                checkpoint = total
    edges = np.memmap(edge_file, dtype=np.uint32, mode="r").reshape(-1, 2)
    graph = sparse.coo_matrix((np.ones(len(edges), dtype=np.uint32), (edges[:, 0], edges[:, 1])),
                              shape=(len(page_ids), len(page_ids))).tocsr()
    graph.sum_duplicates()
    graph.data = np.ones(len(graph.data), dtype=np.uint8)
    graph.sort_indices()
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        sparse.save_npz(stream, graph, compressed=False)
    temporary.replace(path)
    write_json(output / "edge_stage.json", {"raw_link_rows": total, "selected_rows_before_deduplication": kept,
                                            "edges": int(graph.nnz), "nodes": len(page_ids)})
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path(f"data/raw/enwiki_{SNAPSHOT}"))
    parser.add_argument("--output", type=Path, default=Path("data/graphs/fresh_million"))
    parser.add_argument("--retain", type=Path, default=Path("data/graphs/sample_1100/graph.json"))
    parser.add_argument("--extra", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.extra < 1:
        parser.error("--extra must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.raw / "download_manifest.json").read_text())
    for meta in manifest.values():
        path = args.raw / meta["url"].rsplit("/", 1)[-1]
        if not path.exists() or path.stat().st_size != meta["size"]:
            parser.error(f"Missing/incomplete dump table: {path}; run download_fresh_dump.py first")
    started = time.perf_counter()
    if args.retain.is_dir():
        retained_metadata = json.loads((args.retain / "metadata.json").read_text())
        if retained_metadata["snapshot"] != SNAPSHOT:
            parser.error("Retained graph must use the same fresh dump snapshot")
        original = {"graph_id": retained_metadata["graph_id"],
                    "page_ids": np.load(args.retain / "page_ids.npy", mmap_mode="r")}
    else:
        original = json.loads(args.retain.read_text())
    settings = sample_pages(args.raw, args.output, original, args.extra, 207)
    map_targets(args.raw, args.output)
    graph = build_sparse(args.raw, args.output)
    if graph.shape != (settings["nodes"], settings["nodes"]) or graph.nnz == 0:
        raise ValueError("Sparse graph dimensions or edge count are invalid")
    if np.any(graph.data != 1) or np.any(graph.diagonal() != 0):
        raise ValueError("Unexpected graph weights/self loops")
    digest = hashlib.sha256()
    for array in (np.load(args.output / "page_ids.npy"), graph.indptr, graph.indices):
        digest.update(memoryview(np.ascontiguousarray(array)))
    metadata = {"snapshot": SNAPSHOT, "graph_id": "sha256:"+digest.hexdigest(),
                "nodes": int(graph.shape[0]), "edges": int(graph.nnz),
                "sampling": settings, "input_tables": manifest,
                "seconds": time.perf_counter()-started,
                "scope": "Fresh dated dump; namespace-0 nonredirect articles, resolved local redirect chains; induced uniform sample plus retained IDs. Unit costs. No old graph edges mixed in."}
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
