"""Sparse million-node next-link learning, correction, and matched branching tests."""
import argparse
import copy
from collections import deque
import gzip
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import torch

from codes.api.million_semantic_data import DATA, GRAPH, OUT, FEATURE_VERSION, status
from codes.models.next_link.next_link_experiment import Ranker, evaluate_rank, train_rank, train_raw
from codes.api.sample_dataset import save_json

LABELS = DATA/"ranking_labels"
BUDGETS = [200,2000,10000]


def rows_for_goal(graph, distances, goal, count, seed):
    eligible = np.flatnonzero(np.isfinite(distances) & (distances >= 2) & (np.diff(graph.indptr) >= 2))
    rng = np.random.default_rng(seed+int(goal))
    selected = rng.choice(eligible,size=min(count,len(eligible)),replace=False)
    rows = []
    for current in selected:
        neighbors = graph.indices[graph.indptr[current]:graph.indptr[current+1]]
        optimal = distances[neighbors] == distances[current]-1
        if not optimal.any(): raise ValueError("No optimal outgoing edge for reachable decision")
        rows.append({"current":int(current),"goal":int(goal),"candidates":neighbors.astype(int).tolist(),
                     "optimal":optimal.tolist(),"distance":int(distances[current])})
    return rows


def prepare_labels(graph_path=GRAPH, output=LABELS, counts=(64,16,16), samples=(128,64,128), seed=730):
    output.mkdir(parents=True,exist_ok=True)
    metadata = json.loads((graph_path/"metadata.json").read_text())
    config = {"graph_id":metadata["graph_id"],"counts":list(counts),"samples_per_destination":list(samples),"seed":seed}
    if (output/"complete.json").exists():
        if json.loads((output/"complete.json").read_text())["config"] != config: raise ValueError("Label config mismatch")
        return
    if (output/"config.json").exists() and json.loads((output/"config.json").read_text()) != config:
        raise ValueError("Label cache config differs")
    save_json(output/"config.json",config)
    graph = sparse.load_npz(graph_path/"graph.csr.npz").tocsr();reverse = graph.transpose().tocsr()
    ids = np.load(graph_path/"page_ids.npy")
    # Sample destinations from the added million, outside all old 1100 nodes.
    old = Path("data/graphs/sample_1100/graph.json")
    excluded = set(n["pageid"] for n in json.loads(old.read_text())["nodes"]) if old.exists() else set()
    eligible = np.flatnonzero(~np.isin(ids,list(excluded)))
    candidates = np.random.default_rng(seed).permutation(eligible)
    accepted, distances_for_training = [], []
    cache = output/"goals";cache.mkdir(exist_ok=True)
    for goal in candidates:
        path = cache/f"{int(ids[goal])}.npz"
        if path.exists():
            saved = np.load(path)
            if str(saved["graph_id"]) != metadata["graph_id"]: raise ValueError("Cached goal graph differs")
            compact = saved["distances"]
            distances = np.where(compact < 0,np.inf,compact).astype(float)
        else:
            distances = shortest_path(reverse,directed=True,indices=int(goal),unweighted=True)
            finite = np.isfinite(distances)
            if finite.any() and distances[finite].max() >= 32767: raise ValueError("Distances exceed int16 storage")
            compact = np.full(len(ids),-1,dtype=np.int16);compact[finite] = distances[finite].astype(np.int16)
            with path.open("wb") as stream: np.savez_compressed(stream,distances=compact,graph_id=np.array(metadata["graph_id"]))
        group = 0 if len(accepted) < counts[0] else 1 if len(accepted) < sum(counts[:2]) else 2
        rows = rows_for_goal(graph,distances,int(goal),samples[group],seed)
        if len(rows) < samples[group]: continue
        accepted.append((int(goal),rows))
        save_json(output/"progress.json",{"accepted_destinations":len(accepted),"required":sum(counts),"latest_page_id":int(ids[goal])})
        print(f"Exact ranking labels: {len(accepted)}/{sum(counts)} destinations",flush=True)
        if len(accepted) == sum(counts): break
    if len(accepted) != sum(counts): raise ValueError("Insufficient eligible destinations")
    partitions, cursor = {},0
    for name,count in zip(["train","validation","test"],counts):
        goals = accepted[cursor:cursor+count];cursor += count
        rows = [r for _,group in goals for r in group]
        save_json(output/f"{name}.json",rows)
        partitions[name] = {"goal_indices":[g for g,_ in goals],"goal_page_ids":[int(ids[g]) for g,_ in goals],"decisions":len(rows)}
    goal_sets = [set(p["goal_indices"]) for p in partitions.values()]
    assert not any(a&b for i,a in enumerate(goal_sets) for b in goal_sets[i+1:])
    save_json(output/"complete.json",{"config":config,"splits":partitions,
        "scope":"Exact candidate distances in the saved directed million graph; destination-disjoint splits. Only a sample of decisions is supervised, not every article pair."})


def raw_pairs(rows, ids, maximum=60000):
    # Bound regression materialization while drawing candidates from the SAME decisions.
    pairs = {}
    grouped = {}
    for row in rows: grouped.setdefault(row["goal"],[]).append(row)
    for goal,decisions in grouped.items():
        saved = np.load(LABELS/"goals"/f"{int(ids[goal])}.npz")
        distances = saved["distances"]
        for row in decisions:
            for b in row["candidates"]:
                if distances[b] >= 0: pairs[b,goal] = int(distances[b])
    # Trainer supplies the deterministic shuffle/cap. Validation is similarly bounded.
    if maximum is not None and len(pairs) > maximum:
        keys = random.Random(731).sample(sorted(pairs),maximum)
        pairs = {k:pairs[k] for k in keys}
    return pairs


def train_models(seeds=(207,208,209), epochs=10, correction_epochs=5):
    metadata = json.loads((GRAPH/"metadata.json").read_text())
    features = json.loads((DATA/"features/complete.json").read_text())
    if features["graph_id"] != metadata["graph_id"] or features["feature_version"] != FEATURE_VERSION:
        raise ValueError("Feature graph/version mismatch")
    labels = json.loads((LABELS/"complete.json").read_text())
    if labels["config"]["graph_id"] != metadata["graph_id"]: raise ValueError("Label graph mismatch")
    train = json.loads((LABELS/"train.json").read_text());val = json.loads((LABELS/"validation.json").read_text())
    title = torch.from_numpy(np.load(DATA/"features/title.npy",mmap_mode="c"))
    semantic = torch.from_numpy(np.load(DATA/"features/semantic.npy",mmap_mode="c"))
    ids = np.load(GRAPH/"page_ids.npy")
    pairs,val_pairs = raw_pairs(train,ids),raw_pairs(val,ids)
    for seed in seeds:
        folder = OUT/f"seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
        if (folder/"training_complete.json").exists():
            previous = json.loads((folder/"training_complete.json").read_text())
            if previous["graph_id"] != metadata["graph_id"] or previous["epochs"] != epochs or previous["correction_epochs"] != correction_epochs:
                raise ValueError("Completed training config mismatch")
            continue
        torch.manual_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        status("training_raw_distance",seed=seed)
        state,info = train_raw(pairs,val_pairs,title,epochs,seed)
        torch.save({"state_dict":state,"dimensions":128,"hidden":64,"graph_id":metadata["graph_id"]},folder/"raw.pt")
        save_json(folder/"raw.history.json",info)
        for name,use_semantic in [("title_rank",False),("semantic_rank",True)]:
            status("training_next_link",seed=seed,model=name)
            model = Ranker(state,use_semantic)
            train_rank(model,train,val,title,semantic,epochs,seed,folder/f"{name}.pt",graph_id=metadata["graph_id"])
            if use_semantic: initial = model
        metrics,mistakes = evaluate_rank(initial,train,title,semantic,errors=True)
        weights = torch.ones(len(train));weights[mistakes] = 4.
        save_json(folder/"correction_examples.json",{"scope":"Training decisions only", "indices":mistakes,"weight":4,"before":metrics})
        for name,w in [("continued",None),("corrected",weights)]:
            status("training_correction_control",seed=seed,model=name,training_mistakes=len(mistakes))
            model = copy.deepcopy(initial)
            train_rank(model,train,val,title,semantic,correction_epochs,seed+1000,folder/f"{name}.pt",w,graph_id=metadata["graph_id"])
        save_json(folder/"training_complete.json",{"graph_id":metadata["graph_id"],"feature_version":FEATURE_VERSION,
                  "epochs":epochs,"correction_epochs":correction_epochs,"seed":seed,"raw_training_pairs":len(pairs)})


def load_policy(folder,name,graph_id):
    from codes.models.raw.ml_model import DistanceModel
    checkpoint = torch.load(folder/f"{name}.pt",map_location="cpu",weights_only=True)
    if checkpoint["graph_id"] != graph_id: raise ValueError("Checkpoint graph mismatch")
    if name == "raw": model = Ranker(checkpoint["state_dict"])
    else:
        model = Ranker(DistanceModel().state_dict(),checkpoint["semantic"])
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def branching_search(graph,source,goal,model,title,semantic,budgets=BUDGETS,seconds=3.):
    """One fixed search prefix supplies every expansion-budget observation.

    Raw candidate scores prioritize the frontier; no per-neighborhood softmax
    probabilities are compared across different parents. Only BFS is optimal.
    """
    started = time.perf_counter();counter = itertools.count()
    parents = {source:None};expanded,scored = 0,0
    queue = deque([source]) if model is None else [(0.,next(counter),source)]
    records = {};path = None;reason = "frontier_exhausted"
    while queue:
        if time.perf_counter()-started >= seconds: reason = "time_limit";break
        if expanded >= max(budgets): reason = "expansion_budget";break
        node = queue.popleft() if model is None else heapq.heappop(queue)[2]
        if node == goal: reason = "found";break
        expanded += 1
        neighbors = [int(b) for b in graph.indices[graph.indptr[node]:graph.indptr[node+1]] if int(b) not in parents]
        if goal in neighbors:
            parents[goal] = node;reason = "found";break
        if model is None:
            for b in neighbors: parents[b] = node;queue.append(b)
        elif neighbors:
            candidates = torch.tensor(neighbors);goals = torch.full_like(candidates,goal)
            values = []
            for start in range(0,len(candidates),1024):
                if time.perf_counter()-started >= seconds: reason = "time_limit";break
                values.extend(model(candidates[start:start+1024],goals[start:start+1024],title,semantic).tolist())
            if len(values) != len(neighbors): break
            scored += len(neighbors)
            for b,value in zip(neighbors,values):
                parents[b] = node;heapq.heappush(queue,(-value,next(counter),b))
        if expanded in budgets:
            records[str(expanded)] = {"path":None,"clicks":None,"expanded":expanded,"candidates_scored":scored,
                                      "seconds":time.perf_counter()-started,"status":"expansion_budget"}
    elapsed = time.perf_counter()-started
    if elapsed > seconds: reason = "time_limit"
    if reason == "found":
        path = [goal]
        while path[-1] != source: path.append(parents[path[-1]])
        path.reverse()
    final = {"path":path,"clicks":len(path)-1 if path else None,"expanded":expanded,"candidates_scored":scored,"seconds":elapsed,"status":reason}
    for budget in budgets:
        if str(budget) not in records: records[str(budget)] = final.copy()
    return records


@torch.inference_mode()
def greedy(graph,source,goal,model,title,semantic,steps=20):
    path,seen = [source],{source};started = time.perf_counter();scored = 0
    reason = "step_limit"
    for _ in range(steps):
        node = path[-1]
        neighbors = [int(b) for b in graph.indices[graph.indptr[node]:graph.indptr[node+1]] if int(b) not in seen]
        if goal in neighbors: path.append(goal);reason="found";break
        if not neighbors: reason="dead_end";break
        c = torch.tensor(neighbors);d = torch.full_like(c,goal)
        values = torch.cat([model(c[i:i+1024],d[i:i+1024],title,semantic) for i in range(0,len(c),1024)])
        chosen = neighbors[int(values.argmax())];scored += len(neighbors)
        path.append(chosen);seen.add(chosen)
    return {"path":path if reason == "found" else None,"clicks":len(path)-1 if reason == "found" else None,
            "expanded":len(path)-1,"candidates_scored":scored,"seconds":time.perf_counter()-started,"status":reason}


def summarize(rows):
    successful = [r for r in rows if r["path"] is not None]
    return {"queries":len(rows),"successes":len(successful),"optimal_routes":sum(r["clicks"]==r["optimal_clicks"] for r in successful),
            "success_rate":len(successful)/len(rows),"optimal_rate":sum(r["clicks"]==r["optimal_clicks"] for r in successful)/len(rows),
            "mean_expanded_all":float(np.mean([r["expanded"] for r in rows])),
            "mean_candidates_scored_all":float(np.mean([r["candidates_scored"] for r in rows])),
            "median_ms":float(np.median([r["seconds"] for r in rows]))*1000,
            "mean_extra_clicks_successes_only":float(np.mean([r["clicks"]-r["optimal_clicks"] for r in successful])) if successful else None}


def evaluate(seeds=(207,208,209),query_count=96,seconds=3.):
    meta = json.loads((GRAPH/"metadata.json").read_text())
    graph = sparse.load_npz(GRAPH/"graph.csr.npz").tocsr()
    title = torch.from_numpy(np.load(DATA/"features/title.npy",mmap_mode="c"))
    semantic = torch.from_numpy(np.load(DATA/"features/semantic.npy",mmap_mode="c"))
    # Test is loaded after all seeds have completed training and checkpoint selection.
    test = json.loads((LABELS/"test.json").read_text())
    grouped = {}
    for row in test: grouped.setdefault(row["goal"],[]).append(row)
    per_goal = (query_count+len(grouped)-1)//len(grouped)
    queries = [{"source":r["current"],"target":r["goal"],"optimal_clicks":r["distance"]}
               for group in grouped.values() for r in group[:per_goal]][:query_count]
    if len(queries) != query_count: raise ValueError("Insufficient test queries")
    save_json(OUT/"evaluation_queries.json",{"graph_id":meta["graph_id"],"queries":queries})
    # Independently verify all sampled optimal lengths before scoring routes.
    reverse = graph.transpose().tocsr()
    for goal in sorted({q["target"] for q in queries}):
        exact = shortest_path(reverse,directed=True,indices=goal,unweighted=True)
        if any(exact[q["source"]] != q["optimal_clicks"] for q in queries if q["target"] == goal):
            raise ValueError("Independent query-reference check failed")
    del reverse
    titles = {}
    wanted = {q[k] for q in queries for k in ["source","target"]}
    with gzip.open(GRAPH/"nodes.jsonl.gz","rt",encoding="utf-8") as stream:
        for i,line in enumerate(stream):
            if i in wanted: titles[i] = json.loads(line)["title"]
    for q in queries:q.update(source_title=titles[q["source"]],target_title=titles[q["target"]])
    all_results = {}
    bfs = []
    for i,q in enumerate(queries):
        measured = branching_search(graph,q["source"],q["target"],None,title,semantic,seconds=seconds)
        bfs.append({**q,"budgets":measured})
        if (i+1)%12 == 0: status("evaluating_branching",model="bfs",queries_done=i+1,queries=query_count)
    save_json(OUT/"bfs.routes.json",bfs)
    for seed in seeds:
        folder = OUT/f"seed_{seed}";report = {};fingerprints = {}
        for name in ["raw","title_rank","semantic_rank","continued","corrected"]:
            output = folder/f"{name}.evaluation.json"
            if output.exists():
                report[name] = json.loads(output.read_text());continue
            model = load_policy(folder,name,meta["graph_id"])
            signature = hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            if signature in fingerprints:
                # Identical retained weights give identical rankings. Reuse the
                # already measured run; do not present duplicate timings as new measurements.
                prior = fingerprints[signature]
                measured = copy.deepcopy(report[prior]);measured["reused_identical_checkpoint"] = prior
                save_json(output,measured);report[name] = measured;continue
            fingerprints[signature] = name
            with torch.inference_mode(): model(torch.tensor([0]),torch.tensor([0]),title,semantic)
            ranking = evaluate_rank(model,test,title,semantic)
            rows = []
            for i,q in enumerate(queries):
                branch = branching_search(graph,q["source"],q["target"],model,title,semantic,seconds=seconds)
                journey = greedy(graph,q["source"],q["target"],model,title,semantic)
                for r in [journey,*branch.values()]:
                    if r["path"] is not None:
                        if r["clicks"] < q["optimal_clicks"]: raise ValueError("Route shorter than exact reference")
                        if any(not graph[a,b] for a,b in zip(r["path"],r["path"][1:])): raise ValueError("Non-hyperlink move")
                rows.append({**q,"greedy":journey,"budgets":branch})
                if (i+1)%12 == 0: status("evaluating_branching",seed=seed,model=name,queries_done=i+1,queries=query_count)
            save_json(folder/f"{name}.routes.json",rows)
            measured = {"next_link":ranking,"greedy":summarize([{**r,**r["greedy"]} for r in rows]),
                        "branching":{str(b):summarize([{**r,**r["budgets"][str(b)]} for r in rows]) for b in BUDGETS}}
            save_json(output,measured);report[name] = measured
        all_results[str(seed)] = report
    result = {"graph":meta,"features":json.loads((DATA/"features/complete.json").read_text()),
              "labels":json.loads((LABELS/"complete.json").read_text()),"seeds":list(seeds),"queries":len(queries),"seconds_per_query":seconds,
              "budgets":BUDGETS,"bfs":{str(b):summarize([{**r,**r["budgets"][str(b)]} for r in bfs]) for b in BUDGETS},"runs":all_results,
              "protocol":"Same destination-disjoint test decisions and queries for all models. Indexing costs separate; all title/text/category vectors precomputed. Ranking scores guide greedy best-first, not guaranteed optimal. One search prefix supplies budgets 200/2000/10000 with a common 3-second cap. Greedy navigation separately uses 20 clicks. Correction mines only training mistakes; ordinary continuation has equal update count. Identical retained checkpoints reuse measured route outcomes and explicitly marked timings."}
    save_json(OUT/"results.json",result)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=["labels","train","evaluate"])
    a=p.parse_args();torch.set_num_threads(2)
    if a.stage == "labels": prepare_labels()
    elif a.stage == "train": train_models()
    else: evaluate()
