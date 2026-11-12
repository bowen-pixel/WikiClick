"""Show local bulk-data and model progress without contacting Wikipedia."""
import json

from codes.api.million_semantic_data import DATA, OUT, RAW


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    manifest=read(DATA/"download_manifest.json")
    files=[p for p in RAW.glob("*") if p.name.endswith(".bz2") or p.name.endswith(".bz2.part")]
    completed=sum(p.stat().st_size for p in files)
    total=manifest["total_bytes"] if manifest else None
    processed=0
    if manifest:
        for name,info in manifest["files"].items():
            proof=read(DATA/"sources"/(name+".jsonl.gz.manifest.json"))
            if proof and proof.get("source_sha1")==info["sha1"]:
                processed+=info["size"]
    print("Workflow:",json.dumps(read(OUT/"workflow.json"),indent=2))
    print("Current stage:",json.dumps(read(OUT/"status.json"),indent=2))
    print(f"Bulk archives/partials currently on disk: {completed/1e9:.3f} GB")
    if total:
        print(f"Verified, extracted source data: {processed/1e9:.3f} / {total/1e9:.3f} GB (processed archives may be discarded)")
    print("Completed extracted shards:",len(list((DATA/"sources").glob("*.jsonl.gz"))))
    labels=read(DATA/"ranking_labels/complete.json")
    if labels:print("Exact labelled decisions:",{k:v["decisions"] for k,v in labels["splits"].items()})
    else:print("Label generation:",read(DATA/"ranking_labels/progress.json"))
    print("Measured final results available:",(OUT/"results.json").exists())


if __name__=="__main__":main()
