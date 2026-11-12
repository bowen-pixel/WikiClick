"""Resumable bulk-data -> features -> labels -> ranking -> correction -> plots."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from codes.api.million_semantic_data import DATA, OUT, NETWORK_PAUSED_EXIT
from codes.api.sample_dataset import save_json
from codes.paths import ROOT, module_command


def pause():
    """Stop only the recorded controller and its identified stage worker."""
    work=ROOT;os.chdir(work)
    path=OUT/"workflow.json"
    if not path.exists():
        print("No million-ranking workflow has been started.");return
    workflow=json.loads(path.read_text())
    if workflow.get("state") in ("paused","complete"):
        print(f"Workflow is already {workflow['state']}.");return
    pid=workflow.get("worker_pid")
    if not isinstance(pid,int) or pid<=1:raise RuntimeError("No valid controller PID recorded")
    listing=subprocess.check_output(["ps","-axo","pid=,ppid=,command="],text=True)
    processes={}
    for line in listing.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3:processes[int(fields[0])]=(int(fields[1]),fields[2])
    def script(command):
        args=shlex.split(command)
        if len(args)>2 and args[1]=="-m":
            return args[2].rsplit(".",1)[-1]+".py"
        return Path(args[1]).name if len(args)>1 else ""
    if pid not in processes or script(processes[pid][1])!="run_million_ranking.py":
        raise RuntimeError("Recorded controller is absent or its identity changed; no processes were stopped")
    allowed={"million_semantic_data.py","million_next_link.py","plot_million_ranking.py"}
    children=[child for child,(parent,command) in processes.items() if parent==pid and script(command) in allowed]
    # Stop the parent first so child termination cannot trigger another stage.
    for target in [pid,*children]:
        try:os.kill(target,signal.SIGTERM)
        except ProcessLookupError:pass
    time.sleep(.3)
    for name in ["workflow.json","status.json"]:
        file=OUT/name
        value=json.loads(file.read_text()) if file.exists() else {}
        value.update(state="paused",paused_at=datetime.now(timezone.utc).isoformat(),
                     previous_worker_pid=value.get("worker_pid"),worker_pid=None,
                     pause_reason="User requested pause; cached data and partial downloads retained")
        save_json(file,value)
    print("Paused controller and download/processing worker. Saved data can resume later.")


def run(wait_for_collection=False):
    work=ROOT;os.chdir(work)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/"workflow.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=datetime.now(timezone.utc).isoformat()
    os.environ.setdefault("MPLCONFIGDIR",str(work/OUT/".mpl-cache"))
    stage="starting"
    def update(state="running",**details):
        save_json(OUT/"workflow.json",{"state":state,"stage":stage,"worker_pid":os.getpid(),"started_at":started,
                  "updated_at":datetime.now(timezone.utc).isoformat(),**details})
    try:
        if wait_for_collection:
            stage="waiting_for_existing_text_collection";update()
            while not (DATA/"collection_complete.json").exists():
                # Download owns the detailed status file, including current bytes.
                if (OUT/"collection_failed.json").exists():raise RuntimeError("Existing collection failed; see collection_failed.json")
                time.sleep(15)
        stages=[] if (DATA/"collection_complete.json").exists() else [("collecting_bulk_text",["million_semantic_data.py","collect"])]
        stages += [("preparing_semantic_features",["million_semantic_data.py","features"]),
                   ("generating_exact_labels",["million_next_link.py","labels"]),
                   ("training_ranking_and_correction",["million_next_link.py","train"]),
                   ("evaluating_shared_routes",["million_next_link.py","evaluate"]),
                   ("plotting_measured_results",["plot_million_ranking.py"])]
        for stage,command in stages:
            update();print(stage,flush=True)
            subprocess.run(module_command(command),check=True)
        figures="figures/next_link/ranking_million"
        stage="complete";update(state="complete",results=str(OUT/"results.json"),figures=figures)
        save_json(OUT/"status.json",{"state":"complete","results":str(OUT/"results.json"),"figures":figures})
    except subprocess.CalledProcessError as error:
        if error.returncode == NETWORK_PAUSED_EXIT:
            update(state="waiting_for_network",worker_pid=None,
                   message="Reconnect to the internet and rerun .venv/bin/python -m codes.workflows.run_million_ranking; cached progress is retained")
            print("Workflow stopped safely while waiting for network. Resume with the same command once connected.",flush=True)
            return False
        update(state="failed",error=repr(error));raise
    except BaseException as error:
        update(state="failed",error=repr(error));raise
    finally:lock.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--wait-for-collection",action="store_true")
    p.add_argument("--pause",action="store_true",help="Stop the recorded workflow and stage worker, retaining cached data")
    args=p.parse_args()
    result = pause() if args.pause else run(args.wait_for_collection)
    if result is False:raise SystemExit(NETWORK_PAUSED_EXIT)
