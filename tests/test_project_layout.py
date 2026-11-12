"""Regression checks for the consolidated package, launcher, and workflows."""
import json
import subprocess
import sys

from codes.paths import ROOT, module_command


def test_launcher_runs_bfs_from_another_directory(tmp_path):
    dataset = {"pages": [
        {"pageid": i, "title": name, "requested_title": name,
         "url": "https://example.test/" + name, "links": links}
        for i, name, links in [(1, "A", ["B"]), (2, "B", ["C"]), (3, "C", [])]
    ]}
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(dataset))
    result = subprocess.run([sys.executable, str(ROOT / "wikiclick.py"), "bfs",
                             "--data", str(path), "--source", "A", "--target", "C"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "2" in result.stdout
    assert "A" in result.stdout and "B" in result.stdout and "C" in result.stdout


def test_workflow_module_command_resolves_the_moved_stage():
    command = module_command(["million_next_link.py", "labels"])
    result = subprocess.run(command[:3] + ["--help"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert "labels" in result.stdout


def test_pause_recognizes_module_controller_and_its_worker(tmp_path, monkeypatch):
    from codes.workflows import run_million_ranking as workflow
    monkeypatch.setattr(workflow, "OUT", tmp_path)
    (tmp_path / "workflow.json").write_text(json.dumps({"state": "running", "worker_pid": 100}))
    listing = ("100 1 /bin/python -m codes.workflows.run_million_ranking\n"
               "101 100 /bin/python -m codes.models.next_link.million_next_link train\n"
               "102 1 /bin/python -m codes.models.next_link.million_next_link train\n")
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: listing)
    killed = []
    monkeypatch.setattr(workflow.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(workflow.time, "sleep", lambda seconds: None)
    workflow.pause()
    assert killed == [100, 101]
    assert json.loads((tmp_path / "workflow.json").read_text())["state"] == "paused"
