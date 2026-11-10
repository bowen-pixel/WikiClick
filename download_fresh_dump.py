"""Download and checksum the pinned fresh Wikimedia hyperlink tables."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx

from api_endpoint import USER_AGENT

SNAPSHOT = "20261101"
TABLES = ("page", "redirect", "linktarget", "pagelinks")


def sha1(path):
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120) as client:
        response = client.get(f"https://dumps.wikimedia.org/enwiki/{SNAPSHOT}/dumpstatus.json")
        response.raise_for_status()
        files = {name: (job["status"], value) for job in response.json()["jobs"].values()
                 for name, value in job.get("files", {}).items()}
        manifest = {}
        for table in TABLES:
            name = f"enwiki-{SNAPSHOT}-{table}.sql.gz"
            status, meta = files[name]
            if status != "done":
                raise RuntimeError(f"Dump table is not complete: {name}")
            if not meta["url"].startswith(f"/enwiki/{SNAPSHOT}/"):
                raise ValueError("Unexpected dump URL")
            manifest[table] = meta
        (folder / "download_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for table in TABLES:
            meta = manifest[table]
            path = folder / meta["url"].rsplit("/", 1)[-1]
            if path.exists() and path.stat().st_size == meta["size"] and sha1(path) == meta["sha1"]:
                print(f"Verified existing {table}", flush=True)
                continue
            temporary = path.with_suffix(path.suffix+".part")
            for attempt in range(5):
                start = temporary.stat().st_size if temporary.exists() else 0
                try:
                    with client.stream("GET", "https://dumps.wikimedia.org"+meta["url"],
                                       headers={"Range": f"bytes={start}-"} if start else {}) as response:
                        response.raise_for_status()
                        if start and response.status_code != 206:
                            start = 0
                        completed = start
                        checkpoint = completed
                        with temporary.open("ab" if start else "wb") as stream:
                            for block in response.iter_bytes(8*1024*1024):
                                stream.write(block)
                                completed += len(block)
                                if completed-checkpoint >= 256*1024*1024:
                                    print(f"{table}: {completed:,}/{meta['size']:,} bytes", flush=True)
                                    checkpoint = completed
                    break
                except httpx.HTTPError:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
            if temporary.stat().st_size != meta["size"] or sha1(temporary) != meta["sha1"]:
                raise ValueError(f"Size/checksum mismatch for {temporary}; original dataset untouched")
            temporary.replace(path)
            print(f"Downloaded and verified {table}: {meta['size']:,} bytes", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(f"data/raw/enwiki_{SNAPSHOT}"))
    args = parser.parse_args()
    download(args.output)
