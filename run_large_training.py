"""Finish the second-million import, train updated weights, and compare fairly."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sample_dataset import save_json


def run(import_pid=None):
    work = Path(__file__).resolve().parent
    os.chdir(work)
    os.environ.setdefault("MPLCONFIGDIR", str(work/"runs/.mpl-cache"))
    lock = (work/"runs/large_training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = datetime.now(timezone.utc).isoformat()
    status_path = Path("runs/large_training_status.json")
    weights_updated = False
    def update(stage, **details):
        save_json(status_path, {"state": "running", "stage": stage, "worker_pid": os.getpid(),
                               "import_pid": import_pid, "started_at": started,
                               "updated_at": datetime.now(timezone.utc).isoformat(),
                               "target_articles": 2001100, "model_weights_updated": weights_updated, **details})
        print(stage, details, flush=True)
    try:
        if import_pid is not None:
            limit = time.monotonic()+12*3600
            while not Path("data/graphs/fresh_two_million/metadata.json").exists():
                os.kill(import_pid, 0)  # Fail if the producer exited before its atomic metadata commit.
                update("expanding_graph")
                if time.monotonic() > limit:
                    raise TimeoutError("Expanded graph import exceeded 12 hours")
                time.sleep(30)
        else:
            update("expanding_graph")
            subprocess.run([sys.executable, "import_fresh_dump.py", "--retain", "data/graphs/fresh_million",
                            "--extra", "1000000", "--output", "data/graphs/fresh_two_million"], check=True)
        stages = [
            ("generating_exact_training_labels", ["large_training.py", "labels"]),
            ("preparing_title_features", ["large_training.py", "features"]),
            ("training_model_weights", ["large_training.py", "train", "--epochs", "25"]),
            ("evaluating_frozen_control", ["benchmark_fresh_scale.py", "--data", "data/graphs/fresh_two_million",
                "--checkpoint", "runs/model_1100/best.pt", "--query-file", "data/graphs/large_supervised/evaluation_queries.json",
                "--output", "runs/large_frozen_comparison.json"]),
            ("evaluating_trained_model", ["benchmark_fresh_scale.py", "--data", "data/graphs/fresh_two_million",
                "--checkpoint", "runs/model_large_trained/best.pt", "--query-file", "data/graphs/large_supervised/evaluation_queries.json",
                "--output", "runs/large_trained_comparison.json"]),
            ("plotting_frozen_control", ["plot_results.py", "--search-only", "--comparison", "runs/large_frozen_comparison.json",
                "--output", "runs/large_frozen_figures"]),
            ("plotting_trained_model_and_loss", ["plot_results.py", "--training-metrics", "runs/model_large_trained/metrics.json",
                "--comparison", "runs/large_trained_comparison.json", "--output", "runs/large_trained_figures"]),
        ]
        for stage, command in stages:
            update(stage)
            subprocess.run([sys.executable, *command], check=True)
            if stage == "training_model_weights":
                metrics = json.loads(Path("runs/model_large_trained/metrics.json").read_text())
                if metrics["optimizer_updates"] <= 0:
                    raise ValueError("No model weight updates occurred")
                weights_updated = True
        frozen = json.loads(Path("runs/large_frozen_comparison.json").read_text())
        trained = json.loads(Path("runs/large_trained_comparison.json").read_text())
        if [(r["source"], r["target"]) for r in frozen["results"]] != [(r["source"], r["target"]) for r in trained["results"]]:
            raise ValueError("Control and trained comparisons used different queries")
        save_json(status_path, {"state": "complete", "started_at": started,
                               "finished_at": datetime.now(timezone.utc).isoformat(),
                               "model_weights_updated": weights_updated,
                               "checkpoint": "runs/model_large_trained/best.pt",
                               "metrics": "runs/model_large_trained/metrics.json",
                               "diagrams": "runs/large_trained_figures"})
    except BaseException as error:
        save_json(status_path, {"state": "failed", "error": repr(error), "model_weights_updated": weights_updated,
                               "updated_at": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-pid", type=int, help="Wait for an already-running expanded-graph import")
    run(parser.parse_args().import_pid)
