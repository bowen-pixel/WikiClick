"""Verified, resumable bulk text/category features for the existing million graph.

Only namespace-0 page IDs already present in the graph are retained. Categories
are explicit wikitext categories (not template-expanded or filtered for hidden
categories). They are evidence-backed group memberships, not extracted relations.
"""
import bz2
from collections import Counter
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
import zlib

import httpx
import numpy as np

from codes.api.api_endpoint import USER_AGENT
from codes.api.download_fresh_dump import sha1
from codes.models.raw.large_training import fast_title_features
from codes.api.sample_dataset import save_json

GRAPH = Path("data/graphs/fresh_million")
DATA = Path("data/graphs/semantic_million")
RAW = Path("data/raw/enwiki_20261101/semantic_text")
OUT = Path("models/next_link/ranking_million")
FEATURE_VERSION = "title-sha256-128+intro-word-crc32-64+explicit-category-pool-64-v1"
NETWORK_PAUSED_EXIT = 75


def status(stage, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT/"status.json", {"state":"running", "stage":stage, "worker_pid":os.getpid(), "updated_at":time.time(), **details})
    print(stage, json.dumps(details), flush=True)


def select_files(manifest, snapshot="20261101"):
    job = manifest["jobs"]["articlesdump"]
    if job["status"] != "done":
        raise ValueError("Text dump is incomplete")
    files = [(name,meta) for name,meta in job["files"].items() if re.fullmatch(rf"enwiki-{snapshot}-pages-articles\d+\.xml-p\d+p\d+\.bz2",name)]
    if not files:
        raise ValueError("No sharded article text files found")
    for name,meta in files:
        if not meta["url"].startswith(f"/enwiki/{snapshot}/") or meta["size"] <= 0 or not meta.get("sha1"):
            raise ValueError("Invalid text-file manifest")
    return sorted(files, key=lambda item:int(re.search(r"xml-p(\d+)",item[0]).group(1)))


def download_file(client, name, meta):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW/name
    marker = path.with_suffix(path.suffix+".verified.json")
    if path.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved["sha1"] == meta["sha1"] and path.stat().st_size == meta["size"]:
            return path
    if path.exists() and path.stat().st_size == meta["size"] and sha1(path) == meta["sha1"]:
        save_json(marker,{"sha1":meta["sha1"],"size":meta["size"]})
        return path
    partial = path.with_suffix(path.suffix+".part")
    for attempt in range(8):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == meta["size"]:
                break
            if offset > meta["size"]:
                raise ValueError("Partial text file exceeds declared size")
            with client.stream("GET","https://dumps.wikimedia.org"+meta["url"],headers={"Range":f"bytes={offset}-"} if offset else {}) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                if offset and not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected download range")
                completed, checkpoint = offset, offset
                status("downloading_text",file=name,completed_bytes=completed,total_bytes=meta["size"])
                with partial.open("ab" if offset else "wb") as stream:
                    for block in response.iter_bytes(8*1024*1024):
                        stream.write(block);completed += len(block)
                        if completed-checkpoint >= 64*1024*1024:
                            status("downloading_text",file=name,completed_bytes=completed,total_bytes=meta["size"])
                            checkpoint = completed
            break
        except (httpx.HTTPError,OSError) as error:
            if attempt == 7 or (isinstance(error,httpx.ConnectError) and attempt >= 2):
                raise
            print(f"Download request failed ({type(error).__name__}); saved partial retained. Retrying...",flush=True)
            time.sleep(min(30,2**attempt))
    status("verifying_text",file=name)
    if partial.stat().st_size != meta["size"] or sha1(partial) != meta["sha1"]:
        raise ValueError(f"Text checksum mismatch: {partial}; cached originals untouched")
    partial.replace(path)
    save_json(marker,{"sha1":meta["sha1"],"size":meta["size"]})
    return path


def clean_intro(wikitext):
    # Preserve all categories separately before taking the lead. Do not process
    # an entire long article merely to obtain its bounded lead representation.
    text = re.split(r"(?m)^==",wikitext,maxsplit=1)[0][:24000]
    text = re.sub(r"<!--.*?-->"," ",text,flags=re.S)
    text = re.sub(r"<ref\b[^>]*(?:/>|>.*?</ref\s*>)"," ",text,flags=re.S|re.I)
    text = re.sub(r"\{\|.*?\|\}"," ",text,flags=re.S)
    for _ in range(12):
        updated = re.sub(r"\{\{[^{}]*\}\}"," ",text)
        if updated == text: break
        text = updated
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]"," ",text,flags=re.I)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]",r"\2",text)
    text = re.sub(r"\[\[([^\]]+)\]\]",r"\1",text)
    text = re.sub(r"<[^>]+>"," ",text)
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]",r"\1",text)
    return " ".join(html.unescape(text).replace("'''","").replace("''","").split())[:4000]


def text_features(text, dimensions=64):
    tokens = re.findall(r"\w+",text.lower())[:512]
    counts = Counter(tokens)
    result = np.zeros(dimensions,dtype=np.float32)
    for word,count in counts.items():
        digest = zlib.crc32(word.encode("utf-8"))
        result[digest % dimensions] += (1 if digest & 0x80000000 else -1)*count
    norm = np.linalg.norm(result)
    return result/max(1.,norm)


def explicit_categories(text):
    text = re.sub(r"<!--.*?-->","",text,flags=re.S)
    text = re.sub(r"<nowiki\b[^>]*>.*?</nowiki\s*>","",text,flags=re.S|re.I)
    return sorted(set(html.unescape(c).replace("_"," ").strip() for c in
                      re.findall(r"\[\[\s*Category\s*:\s*([^|\]#]+)",text,flags=re.I)))


def extract_shard(path, selected):
    folder = DATA/"sources";folder.mkdir(parents=True,exist_ok=True)
    output = folder/(path.name+".jsonl.gz")
    if output.exists(): return output
    temporary = output.with_suffix(output.suffix+".tmp")
    scanned, kept = 0,0
    status("extracting_text",file=path.name)
    with bz2.open(path,"rb") as stream, gzip.open(temporary,"wt",encoding="utf-8") as target:
        context = ET.iterparse(stream,events=("start","end"))
        _, root = next(context)
        namespace = root.tag.rsplit("}",1)[0]+"}" if "}" in root.tag else ""
        for event,element in context:
            if event != "end" or element.tag != namespace+"page": continue
            scanned += 1
            pageid = int(element.findtext(namespace+"id"))
            if pageid in selected and element.findtext(namespace+"ns") == "0":
                revision = element.find(namespace+"revision")
                text = revision.findtext(namespace+"text",default="") if revision is not None else ""
                # Newer export schemas may put content under slots.
                if not text and revision is not None:
                    text = revision.findtext(".//"+namespace+"text",default="")
                categories = explicit_categories(text)
                row = {"pageid":pageid,"title":element.findtext(namespace+"title"),
                       "revision":revision.findtext(namespace+"id") if revision is not None else None,
                       "timestamp":revision.findtext(namespace+"timestamp") if revision is not None else None,
                       "intro":clean_intro(text),"categories":categories,"dump_file":path.name}
                target.write(json.dumps(row,ensure_ascii=False)+"\n");kept += 1
            element.clear();root.clear()
            if scanned % 100000 == 0: status("extracting_text",file=path.name,pages_scanned=scanned,selected_pages=kept)
    temporary.replace(output)
    save_json(output.with_suffix(output.suffix+".manifest.json"),{"pages_scanned":scanned,"selected_pages":kept})
    return output


def extraction_is_verified(name, meta, graph_id):
    output = DATA/"sources"/(name+".jsonl.gz")
    manifest = output.with_suffix(output.suffix+".manifest.json")
    if not output.exists() or not manifest.exists(): return False
    proof = json.loads(manifest.read_text())
    if proof.get("source_sha1") != meta["sha1"] or proof.get("graph_id") != graph_id: return False
    # Reading to EOF verifies the compressed retained data's CRC, before a
    # resumable run skips downloading its original source again.
    with gzip.open(output,"rt",encoding="utf-8") as stream:
        count = sum(1 for _ in stream)
    return count == proof["selected_pages"]


def retain_extraction_and_remove_raw(path, meta, graph_id):
    """Discard only a reproducible bulk archive after verifying retained records."""
    output = DATA/"sources"/(path.name+".jsonl.gz")
    manifest = output.with_suffix(output.suffix+".manifest.json")
    if not output.exists() or not manifest.exists(): raise ValueError("No completed extraction to retain")
    if not extraction_is_verified(path.name,meta,graph_id):
        # Also migrates the first archive extracted before compact mode existed.
        if path.stat().st_size != meta["size"] or sha1(path) != meta["sha1"]:
            raise ValueError("Refusing to discard an unverified source")
        proof = json.loads(manifest.read_text())
        with gzip.open(output,"rt",encoding="utf-8") as stream:
            count = sum(1 for _ in stream)
        if count != proof["selected_pages"]: raise ValueError("Incomplete retained extraction")
        proof.update(source_sha1=meta["sha1"],source_size=meta["size"],graph_id=graph_id,
                     retained_format="gzip JSONL; bulk source discarded after validation")
        save_json(manifest,proof)
    if path.exists(): path.unlink()
    marker = path.with_suffix(path.suffix+".verified.json")
    if marker.exists(): marker.unlink()


def compact_completed():
    """Offline cleanup only. Never starts a download or resumes the workflow."""
    meta = json.loads((DATA/"download_manifest.json").read_text())
    removed = 0
    for name,info in meta["files"].items():
        path = RAW/name
        output = DATA/"sources"/(name+".jsonl.gz")
        if path.exists() and output.exists():
            size = path.stat().st_size
            retain_extraction_and_remove_raw(path,info,meta["graph_id"])
            removed += size
    print(f"Freed {removed/1e6:.1f} MB of verified bulk archives; retained compressed selected records and partial downloads.",flush=True)


def collection_files(client, meta):
    """A pinned local manifest is enough to resume; no DNS request is needed."""
    path = DATA/"download_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["graph_id"] != meta["graph_id"] or previous["snapshot"] != meta["snapshot"]:
            raise ValueError("Existing text collection belongs to another graph/snapshot")
        files = select_files({"jobs":{"articlesdump":{"status":"done","files":previous["files"]}}},meta["snapshot"])
        if previous["total_bytes"] != sum(m["size"] for _,m in files):
            raise ValueError("Cached text manifest is incomplete")
        print("Using pinned local text manifest; no manifest download required.",flush=True)
        return files
    for attempt in range(3):
        try:
            response = client.get(f"https://dumps.wikimedia.org/enwiki/{meta['snapshot']}/dumpstatus.json")
            response.raise_for_status()
            files = select_files(response.json(),meta["snapshot"])
            save_json(path,{"snapshot":meta["snapshot"],"graph_id":meta["graph_id"],
                          "total_bytes":sum(m["size"] for _,m in files),"files":dict(files)})
            return files
        except httpx.TransportError:
            if attempt == 2: raise
            time.sleep(2**(attempt+1))


def record_network_pause(error):
    file = OUT/"status.json"
    value = json.loads(file.read_text()) if file.exists() else {}
    value.update(state="waiting_for_network",worker_pid=None,error=f"{type(error).__name__}: {error}",
                 resume_command=".venv/bin/python -m codes.workflows.run_million_ranking",updated_at=time.time())
    save_json(file,value)
    save_json(OUT/"collection_paused.json",{"error":str(error),"time":time.time(),"reason":"network"})
    print("Download paused: a network/DNS request failed. Saved extractions and partial downloads are retained.\n"
          "Reconnect to the internet, then run: .venv/bin/python -m codes.workflows.run_million_ranking",file=sys.stderr,flush=True)


def collect():
    DATA.mkdir(parents=True,exist_ok=True)
    meta = json.loads((GRAPH/"metadata.json").read_text())
    for name in ["collection_failed.json","collection_paused.json"]:
        marker = OUT/name
        if marker.exists(): marker.unlink()
    with httpx.Client(headers={"User-Agent":USER_AGENT},timeout=120,follow_redirects=True) as client:
        files = collection_files(client,meta)
        selected = set(map(int,np.load(GRAPH/"page_ids.npy")))
        for position,(name,info) in enumerate(files):
            if extraction_is_verified(name,info,meta["graph_id"]):
                if (RAW/name).exists(): retain_extraction_and_remove_raw(RAW/name,info,meta["graph_id"])
                print(f"Reusing verified retained extraction: {name}",flush=True)
                continue
            status("text_shard",shard=position+1,shards=len(files),total_download_bytes=sum(m["size"] for _,m in files))
            path = download_file(client,name,info)
            extract_shard(path,selected)
            retain_extraction_and_remove_raw(path,info,meta["graph_id"])
    save_json(DATA/"collection_complete.json",{"graph_id":meta["graph_id"],"shards":len(files)})


def features():
    DATA.mkdir(parents=True,exist_ok=True)
    meta = json.loads((GRAPH/"metadata.json").read_text())
    ids = np.load(GRAPH/"page_ids.npy")
    if not np.all(ids[:-1] < ids[1:]): raise ValueError("Page IDs must be sorted")
    n = len(ids)
    cache = DATA/"features";cache.mkdir(exist_ok=True)
    if (cache/"complete.json").exists():
        previous = json.loads((cache/"complete.json").read_text())
        if previous["graph_id"] != meta["graph_id"] or previous["feature_version"] != FEATURE_VERSION:
            raise ValueError("Feature cache mismatch")
        return
    title = np.lib.format.open_memmap(cache/"title.npy",mode="w+",dtype=np.float32,shape=(n,128))
    text = np.lib.format.open_memmap(cache/"text.npy",mode="w+",dtype=np.float32,shape=(n,64));text[:] = 0
    found = np.zeros(n,dtype=bool)
    status("title_features",nodes=n)
    with gzip.open(GRAPH/"nodes.jsonl.gz","rt",encoding="utf-8") as stream:
        for i,line in enumerate(stream):
            row = json.loads(line)
            if i >= n or row["pageid"] != ids[i]: raise ValueError("Title/index order differs from graph")
            title[i] = fast_title_features(row["title"],128)
            if (i+1) % 100000 == 0: status("title_features",completed_nodes=i+1,nodes=n)
    if i+1 != n: raise ValueError("Incomplete title table")
    title.flush()
    categories = {}
    for position,path in enumerate(sorted((DATA/"sources").glob("*.jsonl.gz"))):
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line); i = int(np.searchsorted(ids,row["pageid"]))
                if i == n or ids[i] != row["pageid"]: raise ValueError("Unexpected extracted page")
                text[i] = text_features(row["intro"]);found[i] = True
                for category in row["categories"]: categories.setdefault(category,[]).append(i)
        status("text_features",shards_processed=position+1,articles_found=int(found.sum()))
    text.flush()
    if found.mean() < .95:
        raise ValueError(f"Only {int(found.sum())}/{n} text articles found; refusing semantic training on incomplete collection")
    # A hyperedge membership list stays sparse. No clique expansion or dense NxN matrix.
    groups = {c:np.asarray(sorted(set(m)),dtype=np.int32) for c,m in categories.items() if 2 <= len(m) <= 1000}
    del categories
    pooled = np.lib.format.open_memmap(cache/"semantic.npy",mode="w+",dtype=np.float32,shape=(n,128))
    pooled[:,:64] = text; pooled[:,64:] = 0
    degree = np.zeros(n,dtype=np.float32)
    with gzip.open(DATA/"category_hyperedges.jsonl.gz","wt",encoding="utf-8") as stream:
        for category,members in sorted(groups.items()):
            pooled[members,64:] += np.asarray(text[members]).mean(0);degree[members] += 1
            stream.write(json.dumps({"category":category,"member_indices":members.tolist()},ensure_ascii=False)+"\n")
    for start in range(0,n,8192):
        chunk = pooled[start:start+8192,64:]
        chunk /= degree[start:start+8192,None].clip(1)
        norms = np.linalg.norm(chunk,axis=1,keepdims=True).clip(1e-12)
        chunk /= norms
    pooled.flush()
    np.save(cache/"text_found.npy",found)
    summary = {"graph_id":meta["graph_id"],"snapshot":meta["snapshot"],"feature_version":FEATURE_VERSION,
               "nodes":n,"text_articles_found":int(found.sum()),"category_hyperedges":len(groups),
               "nonempty_intro_vectors":int(np.count_nonzero(np.linalg.norm(text,axis=1))),
               "articles_in_hyperedges":int((degree>0).sum()),
               "scope":"Fixed bounded word-hashed intro features and explicit-category group pooling. Missing text is zero-filled. No pretrained text encoder, learned relation extractor, or learned hypergraph propagation. Categories may include hidden/maintenance categories and omit template-inherited membership. All graph text is visible as transductive unlabeled input."}
    save_json(cache/"complete.json",summary)


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","features","compact"])
    args=p.parse_args()
    try:
        if args.stage == "collect": collect()
        elif args.stage == "features": features()
        else: compact_completed()
    except httpx.TransportError as error:
        if args.stage != "collect": raise
        record_network_pause(error)
        raise SystemExit(NETWORK_PAUSED_EXIT) from None
    except BaseException as error:
        if args.stage == "collect":
            save_json(OUT/"collection_failed.json",{"error":repr(error),"time":time.time()})
        raise
