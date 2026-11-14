# Deploy

Run commands from the cloned WikiClick repository. 

Requires Python 3.13 and an internet connection are used for live Wikipedia navigation. 
This is a terminal application; no hosted web service or GitHub Pages deployment is configured.

## 1. Install

```bash
git clone https://github.com/bowen-pixel/WikiClick.git
cd WikiClick
python3.13 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
```

On Windows use `py -3.13 -m venv .venv` and `.venv\Scripts\python.exe` in place of `.venv/bin/python`. Pins reflect the inspected local environment; installation on other platforms is not verified.

## 2. Download the live encoder

```bash
.venv/bin/python code/run.py download-encoder
```

This explicitly downloads [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), revision `f9972ac22e696a105c482202646979e33e4b5a1e`, including tokenizer/configuration, pooling configuration and model weights. The default destination is `data/cache/semantic_encoders/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e/`. Download access is governed by the upstream model repository. Live inference loads this directory locally and never downloads a replacement silently. The download helper was added in this refactor; the existing local encoder, rather than a new full download, was used for verification.

No WikiClick trained checkpoint is needed for the default **M0 cosine** live search. Historical WikiClick `.pt` files are intentionally local and have no published release download URL. Do not assume that cloning provides them.

## 3. Enter two URLs

```bash
.venv/bin/python code/run.py live
```

```text
Source: https://en.wikipedia.org/wiki/Isaac_Newton
Target: https://en.wikipedia.org/wiki/Apple
```

Press Enter after each link. The API checks existence, namespace and redirects. Invalid URL formats prompt again. Missing articles and API errors are reported. A successful run displays an arrow pathway, a numbered URL for every step, click count and query count. Links can change after collection; a route is a candidate shortest path, not a global optimality certificate.

```bash
.venv/bin/python code/run.py live \
  --source 'https://en.wikipedia.org/wiki/FC_Barcelona' \
  --target 'https://en.wikipedia.org/wiki/Gao_Qiu' \
  --max-expansions 30 --max-clicks 10 --max-api-queries 200 \
  --output logs/live_route.json
```

Budgets bound work rather than guarantee success or wall-clock duration. Direct/two-hop probes count against API queries. The expansion budget applies to semantic search. CPU encoding may take minutes on large frontiers. API retries can add HTTP requests. Ctrl+C cancels the run. A failed bounded search does not establish that no route exists.

For a different local encoder location, use `download-encoder --output PATH`, then `live --encoder PATH`. A compatible `navigation/scaling` M1/M2 checkpoint can be supplied with `live --checkpoint PATH`; it must use the matching 128-dimensional title-plus-introduction representation. Graph-bound SHA256 rankers are not interchangeable with these checkpoints.

## 4. Try BFS without encoder weights

The consolidated JSON is already included under `training_data/`; it is approximately 86 MB and can be downloaded with the repository. Restore the original small graph and labels:

```bash
.venv/bin/python code/run.py prepare-data --dataset sample_100
.venv/bin/python code/run.py bfs --data data/graphs/sample_100/articles.json
```

BFS prompts for titles from the displayed saved article list. This is an offline shortest-path demo. Restoration preserves existing local files rather than overwriting them. The public bundle contains the small graph's full topology and labelled splits; it does not restore the full million-node topology or frozen features.

## 5. Train the distance network

```bash
.venv/bin/python code/run.py raw-train \
  --data data/graphs/sample_100 --features sha256 \
  --epochs 25 --output models/raw/my_distance_model
```

Training writes ignored local checkpoints/results. Existing checkpoints are protected by the trainer. `--features semantic` uses the pretrained title encoder instead; it can contact the upstream model repository. Feature choices, seeds and dataset identities must be recorded before comparing results.

## 6. Train and run the original next-link ranker

```bash
.venv/bin/python code/run.py prepare-data --dataset ranking_1000
.venv/bin/python code/run.py next-link-train fetch
.venv/bin/python code/run.py next-link-train run
.venv/bin/python code/run.py next-link
```

`fetch` explicitly collects current introductions/categories. New metadata may differ from the historical experiment, so newly trained results are not a bit-for-bit reproduction of published statistics. The trainer writes original-format local model bundles in `models/next_link/ranking_1000/`; use a fresh copy if you wish to preserve earlier checkpoints. Saved navigation accepts titles in that graph and never updates weights during inference. `evolve --help` exposes the architecture-search core and requires the trained next-link model/data bundle. Research-scale benchmark drivers remain local rather than being prerequisites for running the public code.

## Public versus local files

The public snapshot contains `code/`, `training_data/`, curated `figures/`, this guide and the four root Markdown documents. `.gitignore` enforces the allowlist. Tests, old code, research drivers, logs, virtual environments, local weights, raw dumps and caches remain on the original device. Removing a tracked path from the index preserves its working file; it does not erase Git history. This refactor does not publish a release, upload model weights or rewrite prior commits.
