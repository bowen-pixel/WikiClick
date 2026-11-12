import bz2
import gzip
import json

import numpy as np
from scipy import sparse

import codes.api.million_semantic_data as semantic
from codes.models.next_link.million_next_link import rows_for_goal, branching_search


def small_graph():
    return sparse.csr_matrix((np.ones(5),([0,0,0,1,2],[1,2,4,3,3])),shape=(5,5))


def test_exact_labels_include_all_optimal_neighbors_and_negative():
    distances=np.array([2.,1.,1.,0.,np.inf])
    rows=rows_for_goal(small_graph(),distances,3,10,1)
    assert len(rows)==1
    assert rows[0]["candidates"]==[1,2,4]
    assert rows[0]["optimal"]==[True,True,False]


def test_branching_budget_prefix_and_optimal_bfs():
    rows=branching_search(small_graph(),0,3,None,None,None,budgets=[1,2,10],seconds=10)
    assert rows["1"]["path"] is None
    assert rows["1"]["expanded"]==1
    assert rows["2"]["path"]==[0,1,3]
    assert rows["10"]["clicks"]==2


def test_deadline_failure_has_no_fake_path():
    rows=branching_search(small_graph(),0,3,None,None,None,budgets=[1,2],seconds=0)
    assert all(r["status"]=="time_limit" and r["path"] is None for r in rows.values())


def test_evidence_extraction_and_normalized_text():
    text="{{Infobox|x={{nested}}}} '''A''' is a [[Film|movie]].<ref>citation</ref>\n== Career ==\nrest"
    lead=semantic.clean_intro(text)
    assert "A is a movie." in lead
    assert "citation" not in lead and "nested" not in lead and "Career" not in lead
    cats=semantic.explicit_categories('[[Category:Race_cars|X]] <!-- [[Category:Hidden]] --> <nowiki>[[Category:Example]]</nowiki>')
    assert cats==["Race cars"]
    assert np.isclose(np.linalg.norm(semantic.text_features("racing film actor")),1)
    assert not semantic.text_features("").any()


def test_xml_extraction_uses_page_id_not_revision_id(tmp_path,monkeypatch):
    monkeypatch.setattr(semantic,"DATA",tmp_path/"data")
    monkeypatch.setattr(semantic,"OUT",tmp_path/"out")
    xml='''<mediawiki xmlns="urn:test"><page><title>A</title><ns>0</ns><id>11</id><revision><id>999</id><timestamp>2026-11-01T00:00:00Z</timestamp><text>A is a [[Film]]. [[Category:Films]]</text></revision></page><page><title>B</title><ns>0</ns><id>12</id><revision><id>998</id><text>Other</text></revision></page></mediawiki>'''
    path=tmp_path/"part.bz2";path.write_bytes(bz2.compress(xml.encode()))
    output=semantic.extract_shard(path,{11})
    with gzip.open(output,"rt") as stream: rows=[json.loads(line) for line in stream]
    assert len(rows)==1 and rows[0]["pageid"]==11
    assert rows[0]["revision"]=="999" and rows[0]["categories"]==["Films"]
    assert semantic.extract_shard(path,{11})==output


def test_sparse_category_pooling_full_pipeline(tmp_path,monkeypatch):
    graph=tmp_path/"graph";data=tmp_path/"data";graph.mkdir();(data/"sources").mkdir(parents=True)
    monkeypatch.setattr(semantic,"GRAPH",graph);monkeypatch.setattr(semantic,"DATA",data);monkeypatch.setattr(semantic,"OUT",tmp_path/"out")
    (graph/"metadata.json").write_text(json.dumps({"graph_id":"test","snapshot":"20261101"}))
    np.save(graph/"page_ids.npy",np.array([11,12]))
    with gzip.open(graph/"nodes.jsonl.gz","wt") as stream:
        for i in [11,12]:stream.write(json.dumps({"pageid":i,"title":str(i)})+"\n")
    with gzip.open(data/"sources/part.jsonl.gz","wt") as stream:
        for i,text in [(11,"racing"),(12,"film")]:stream.write(json.dumps({"pageid":i,"intro":text,"categories":["Shared"]})+"\n")
    semantic.features()
    vectors=np.load(data/"features/semantic.npy")
    assert vectors.shape==(2,128)
    assert np.allclose(vectors[0,64:],vectors[1,64:])
    manifest=json.loads((data/"features/complete.json").read_text())
    assert manifest["category_hyperedges"]==1 and manifest["articles_in_hyperedges"]==2


def test_end_to_end_sparse_training_correction_and_shared_evaluation(tmp_path,monkeypatch):
    import torch
    import codes.models.next_link.million_next_link as experiment
    graph=tmp_path/"graph";data=tmp_path/"data";out=tmp_path/"out";labels=data/"ranking_labels"
    graph.mkdir();(data/"sources").mkdir(parents=True)
    for module in [semantic,experiment]:
        monkeypatch.setattr(module,"GRAPH",graph);monkeypatch.setattr(module,"DATA",data);monkeypatch.setattr(module,"OUT",out)
    monkeypatch.setattr(experiment,"LABELS",labels)
    n=12; ids=np.arange(900000000,900000000+n)
    edges=[(i,(i+j)%n) for i in range(n) for j in [1,2]]
    csr=sparse.csr_matrix((np.ones(len(edges)),tuple(zip(*edges))),shape=(n,n))
    sparse.save_npz(graph/"graph.csr.npz",csr);np.save(graph/"page_ids.npy",ids)
    (graph/"metadata.json").write_text(json.dumps({"graph_id":"fixture","snapshot":"20261101","nodes":n,"edges":len(edges)}))
    with gzip.open(graph/"nodes.jsonl.gz","wt") as stream, gzip.open(data/"sources/part.jsonl.gz","wt") as text:
        for i,pageid in enumerate(ids):
            stream.write(json.dumps({"pageid":int(pageid),"title":f"Article {i}"})+"\n")
            text.write(json.dumps({"pageid":int(pageid),"intro":f"Topic {i%2} article","categories":[f"Group {i%2}"]})+"\n")
    semantic.features()
    experiment.prepare_labels(graph,labels,counts=(2,1,1),samples=(3,3,3),seed=2)
    torch.set_num_threads(2)
    experiment.train_models(seeds=(7,),epochs=1,correction_epochs=1)
    experiment.evaluate(seeds=(7,),query_count=3,seconds=1.)
    result=json.loads((out/"results.json").read_text())
    assert result["queries"]==3
    assert set(result["runs"]["7"])=={"raw","title_rank","semantic_rank","continued","corrected"}
    assert result["bfs"]["10000"]["optimal_routes"]==3
    checkpoint=torch.load(out/"seed_7/corrected.pt",weights_only=True)
    assert checkpoint["graph_id"]=="fixture"
    mistakes=json.loads((out/"seed_7/correction_examples.json").read_text())
    assert mistakes["scope"]=="Training decisions only"


def test_compaction_preserves_verified_extraction_and_resume(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.setattr(semantic,"DATA",tmp_path/"data");monkeypatch.setattr(semantic,"OUT",tmp_path/"out")
    xml='<mediawiki><page><title>A</title><ns>0</ns><id>11</id><revision><id>999</id><text>Article text</text></revision></page></mediawiki>'
    path=tmp_path/"part.bz2";raw=bz2.compress(xml.encode());path.write_bytes(raw)
    meta={"size":len(raw),"sha1":hashlib.sha1(raw).hexdigest()}
    output=semantic.extract_shard(path,{11})
    semantic.retain_extraction_and_remove_raw(path,meta,"test")
    assert not path.exists() and output.exists()
    assert semantic.extraction_is_verified("part.bz2",meta,"test")
    assert not semantic.extraction_is_verified("part.bz2",meta,"different-graph")


def test_compaction_refuses_incomplete_retained_data(tmp_path,monkeypatch):
    import hashlib
    import pytest
    monkeypatch.setattr(semantic,"DATA",tmp_path/"data");monkeypatch.setattr(semantic,"OUT",tmp_path/"out")
    xml='<mediawiki><page><title>A</title><ns>0</ns><id>11</id><revision><text>Article</text></revision></page></mediawiki>'
    path=tmp_path/"part.bz2";raw=bz2.compress(xml.encode());path.write_bytes(raw)
    meta={"size":len(raw),"sha1":hashlib.sha1(raw).hexdigest()}
    output=semantic.extract_shard(path,{11})
    with gzip.open(output,"wt") as stream:stream.write("")
    with pytest.raises(ValueError,match="Incomplete retained extraction"):
        semantic.retain_extraction_and_remove_raw(path,meta,"test")
    assert path.exists()


def test_pause_stops_only_controller_and_recognized_child(tmp_path,monkeypatch):
    import codes.workflows.run_million_ranking as workflow
    monkeypatch.setattr(workflow,"OUT",tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path/"workflow.json").write_text(json.dumps({"state":"running","worker_pid":100}))
    (tmp_path/"status.json").write_text(json.dumps({"state":"running","worker_pid":101}))
    listing="100 1 /bin/python -m codes.workflows.run_million_ranking\n101 100 /bin/python -m codes.api.million_semantic_data collect\n102 1 /bin/python unrelated.py\n"
    monkeypatch.setattr(workflow.subprocess,"check_output",lambda *a,**k:listing)
    killed=[]
    monkeypatch.setattr(workflow.os,"kill",lambda pid,sig:killed.append(pid))
    monkeypatch.setattr(workflow.time,"sleep",lambda seconds:None)
    workflow.pause()
    assert killed==[100,101]
    assert json.loads((tmp_path/"workflow.json").read_text())["state"]=="paused"


def test_pause_refuses_reused_pid(tmp_path,monkeypatch):
    import pytest
    import codes.workflows.run_million_ranking as workflow
    monkeypatch.setattr(workflow,"OUT",tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path/"workflow.json").write_text(json.dumps({"state":"running","worker_pid":100}))
    monkeypatch.setattr(workflow.subprocess,"check_output",lambda *a,**k:"100 1 /bin/python unrelated.py\n")
    killed=[];monkeypatch.setattr(workflow.os,"kill",lambda pid,sig:killed.append(pid))
    with pytest.raises(RuntimeError,match="identity changed"):workflow.pause()
    assert not killed


def test_resume_manifest_needs_no_network(tmp_path,monkeypatch):
    import httpx
    monkeypatch.setattr(semantic,"DATA",tmp_path)
    meta={"graph_id":"graph","snapshot":"20261101"}
    filename="enwiki-20261101-pages-articles1.xml-p1p100.bz2"
    record={"size":123,"sha1":"f"*40,"url":"/enwiki/20261101/"+filename}
    (tmp_path/"download_manifest.json").write_text(json.dumps({**meta,"files":{filename:record},"total_bytes":123}))
    def no_request(request):raise AssertionError("Cached manifest must not contact the network")
    with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
        assert semantic.collection_files(client,meta)==[(filename,record)]


def test_dns_failure_keeps_partial_download(tmp_path,monkeypatch):
    import httpx
    import pytest
    monkeypatch.setattr(semantic,"RAW",tmp_path);monkeypatch.setattr(semantic.time,"sleep",lambda n:None)
    name="part.bz2";partial=tmp_path/(name+".part");partial.write_bytes(b"saved")
    calls=[]
    def offline(request):
        calls.append(request)
        raise httpx.ConnectError("DNS lookup failed",request=request)
    with httpx.Client(transport=httpx.MockTransport(offline)) as client:
        with pytest.raises(httpx.ConnectError):
            semantic.download_file(client,name,{"url":"/enwiki/test/part.bz2","size":100,"sha1":"f"*40})
    assert partial.read_bytes()==b"saved" and len(calls)==3
    assert all(r.headers["Range"]=="bytes=5-" for r in calls)


def test_controller_reports_waiting_for_network_without_traceback(tmp_path,monkeypatch):
    import subprocess
    import codes.workflows.run_million_ranking as workflow
    monkeypatch.setattr(workflow,"OUT",tmp_path/"out");monkeypatch.setattr(workflow,"DATA",tmp_path/"data")
    monkeypatch.chdir(tmp_path)
    def offline(*args,**kwargs):raise subprocess.CalledProcessError(semantic.NETWORK_PAUSED_EXIT,args[0])
    monkeypatch.setattr(workflow.subprocess,"run",offline)
    assert workflow.run() is False
    record=json.loads((tmp_path/"out/workflow.json").read_text())
    assert record["state"]=="waiting_for_network" and record["worker_pid"] is None
