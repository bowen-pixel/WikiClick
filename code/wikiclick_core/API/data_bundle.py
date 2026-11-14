"""Restore small-graph training inputs from the single public JSON bundle."""
import argparse
import json
from pathlib import Path
from ..paths import ROOT


def restore(name, root=ROOT):
    bundle = json.loads((root/"training_data/training_data.json").read_text())
    saved = bundle["graphs"][name]
    folder = root/"data/graphs"/name
    folder.mkdir(parents=True, exist_ok=True)
    files = {"graph.json": saved["graph"], "manifest.json": saved["manifest"]}
    for source in bundle["sources"]:
        path = Path(source["path"])
        if path.parent.as_posix() != "data/graphs/"+name:
            continue
        if source["encoding"] == "columns":
            rows = []
            titles = {n["pageid"]: n["title"] for n in saved["graph"]["nodes"]}
            for current, goal, candidate, distance in source["data"]:
                article = lambda n: {"pageid": n, "title": titles[n]}
                rows.append({"graph_id": source["graph_id"], "split": source["split"],
                             "current_article": article(current), "destination": article(goal),
                             "candidate_next_article": article(candidate), "label_distance": distance,
                             "reachable_in_graph": distance is not None})
            files[path.name] = rows
        else:
            files[path.name] = source["data"]
    graph = saved["graph"]
    targets = {n["pageid"]: [] for n in graph["nodes"]}
    for edge in graph["edges"]:
        targets[edge["source"]].append({"pageid": edge["target"]})
    files["articles.json"] = {"pages": [{**n, "requested_title": n["title"],
        "url": n.get("url", "https://en.wikipedia.org/?curid="+str(n["pageid"])),
        "link_targets": targets[n["pageid"]]} for n in graph["nodes"]]}
    for name, data in files.items():
        path = folder/name
        if path.exists():
            print(f"Preserved existing {path}")
            continue
        with path.open("w", encoding="utf-8") as stream:
            if path.suffix == ".jsonl":
                for row in data:
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            else:
                json.dump(data, stream, ensure_ascii=False)
                stream.write("\n")
        print(f"Restored {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sample_100", "sample_1100", "ranking_1000"], default="sample_100")
    args = parser.parse_args()
    restore(args.dataset)


if __name__ == "__main__":
    main()
