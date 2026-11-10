"""Run the long fresh-data import, sampled benchmark, and diagram job."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from download_fresh_dump import SNAPSHOT
from import_fresh_dump import write_json


def run(wait_for_download=False):
    work = Path(__file__).resolve().parent
    os.chdir(work)
    os.environ.setdefault("MPLCONFIGDIR", str(work / "runs/.mpl-cache"))
    raw = Path(f"data/raw/enwiki_{SNAPSHOT}")
    status_file = Path("runs/fresh_scale_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    def status(stage, **details):
        write_json(status_file, {"state": "running", "stage": stage, "pid": os.getpid(),
                                 "started_at": started, "updated_at": datetime.now(timezone.utc).isoformat(),
                                 "requested_extra_articles": 1_000_000, "retained_articles": 1100,
                                 "snapshot": SNAPSHOT, **details})
        print(stage, details, flush=True)

    try:
        if wait_for_download:
            limit = time.monotonic()+24*3600
            while True:
                manifest = json.loads((raw / "download_manifest.json").read_text())
                complete = []
                received = 0
                for name, meta in manifest.items():
                    path = raw / meta["url"].rsplit("/", 1)[-1]
                    partial = path.with_suffix(path.suffix+".part")
                    if path.exists() and path.stat().st_size == meta["size"]:
                        complete.append(name)
                        received += path.stat().st_size
                    elif partial.exists():
                        received += partial.stat().st_size
                status("downloading", completed_tables=complete, downloaded_bytes=received,
                       expected_bytes=sum(meta["size"] for meta in manifest.values()))
                if len(complete) == 4:
                    break
                if time.monotonic() > limit:
                    raise TimeoutError("Download wait exceeded 24 hours; partial files retained")
                time.sleep(30)
        else:
            status("downloading")
            subprocess.run([sys.executable, "download_fresh_dump.py"], check=True)
        for stage, command in [
            ("importing_fresh_graph", ["import_fresh_dump.py"]),
            ("benchmarking_sampled_queries", ["benchmark_fresh_scale.py", "--queries", "100", "--budget", "200", "--seconds-per-query", "10"]),
            ("generating_measured_diagrams", ["plot_results.py", "--search-only", "--comparison", "runs/fresh_million_comparison.json", "--output", "runs/fresh_million_figures"]),
        ]:
            status(stage)
            subprocess.run([sys.executable, *command], check=True)
        write_json(status_file, {"state": "complete", "pid": os.getpid(), "snapshot": SNAPSHOT,
                                 "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                                 "comparison": "runs/fresh_million_comparison.json",
                                 "diagrams": "runs/fresh_million_figures"})
    except BaseException as error:
        write_json(status_file, {"state": "failed", "pid": os.getpid(), "snapshot": SNAPSHOT,
                                 "started_at": started, "error": repr(error),
                                 "updated_at": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-download", action="store_true", help="An existing download process already owns the partial files")
    run(parser.parse_args().wait_for_download)
