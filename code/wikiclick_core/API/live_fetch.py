"""Search live English Wikipedia using the existing semantic best-first engine."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

import httpx

from wikiclick_core.API.wiki_api import USER_AGENT, query_api
from wikiclick_core.paths import ROOT


def article_title(value):
    value = value.strip()
    if "://" in value:
        url = urlsplit(value)
        if (url.scheme not in ("http", "https") or
                url.netloc not in ("en.wikipedia.org", "en.m.wikipedia.org") or
                not url.path.startswith("/wiki/")):
            raise ValueError("Use an English Wikipedia /wiki/ URL or an article title")
        value = unquote(url.path[len("/wiki/"):])
    value = value.replace("_", " ").strip()
    if not value or "|" in value:
        raise ValueError("Provide one article title per endpoint")
    return value


def prompt_endpoint(label):
    """Prompt for a page URL; article existence is checked by the API later."""
    while True:
        value = input(f"{label}: ").strip()
        try:
            if "://" not in value:
                raise ValueError("Paste a full English Wikipedia page link, such as https://en.wikipedia.org/wiki/Isaac_Newton")
            article_title(value)
            return value
        except ValueError as error:
            print(error)


class RequestBudgetExceeded(RuntimeError):
    pass


class LiveGraph:
    """Lazy directed graph; publish an adjacency only after full pagination."""
    def __init__(self, http, max_requests=200):
        self.http = http
        self.max_requests = max_requests
        self.requests = 0
        self.pages = {}
        self.adjacency = {}

    def query(self, params):
        if self.requests >= self.max_requests:
            raise RequestBudgetExceeded("API query budget exhausted")
        self.requests += 1
        return query_api(self.http, params)

    def remember(self, page):
        if ("missing" in page or "invalid" in page or
                "pageid" not in page or page.get("ns") != 0):
            return None
        node = page["pageid"]
        self.pages[node] = {**self.pages.get(node, {}), **page}
        return node

    def resolve(self, value):
        raw = self.query({"titles": article_title(value), "redirects": 1,
                          "prop": "info|extracts", "inprop": "url",
                          "exintro": 1, "explaintext": 1})
        pages = raw.get("query", {}).get("pages", [])
        if len(pages) != 1 or self.remember(pages[0]) is None:
            raise ValueError(f"Not an existing English Wikipedia article: {value}")
        return pages[0]["pageid"]

    def successors(self, node):
        if node in self.adjacency:
            return self.adjacency[node]
        print(f"Fetching links: {self.pages[node]['title']}", file=sys.stderr, flush=True)
        params = {"generator": "links", "pageids": node, "redirects": 1,
                  "gplnamespace": 0, "gpllimit": "max",
                  "prop": "info", "inprop": "url"}
        targets = set()
        while True:
            raw = self.query(params)
            for page in raw.get("query", {}).get("pages", []):
                target = self.remember(page)
                if target is not None and target != node:
                    targets.add(target)
            if "continue" not in raw:
                break
            params.update(raw["continue"])
        self.adjacency[node] = sorted(targets)
        return self.adjacency[node]

    def ensure_texts(self, nodes):
        # Check for a direct target link before spending requests on introductions.
        missing = list(dict.fromkeys(n for n in nodes if "extract" not in self.pages[n]))
        for start in range(0, len(missing), 20):
            batch = missing[start:start + 20]
            raw = self.query({"pageids": "|".join(map(str, batch)),
                              "prop": "extracts", "exintro": 1,
                              "explaintext": 1, "exlimit": 20})
            returned = set()
            for page in raw.get("query", {}).get("pages", []):
                node = self.remember(page)
                if node in batch:
                    self.pages[node]["extract"] = page.get("extract", "")
                    returned.add(node)
            if returned != set(batch):
                raise ValueError("Article metadata changed or is missing during live search")

    def short_route(self, source, target, max_clicks):
        """Check real two-link bridges before encoding a large frontier.

        These narrow edge probes do not fetch complete candidate adjacencies.
        Reserve half the remaining API budget for semantic search if none match.
        """
        if source == target:
            return [source]
        candidates = self.successors(source)
        if target in candidates:
            return [source, target]
        if max_clicks < 2:
            return None
        batches = (self.max_requests - self.requests) // 2
        for start in range(0, min(len(candidates), batches * 50), 50):
            batch = candidates[start:start + 50]
            raw = self.query({"pageids": "|".join(map(str, batch)),
                              "prop": "links", "plnamespace": 0,
                              "pltitles": self.pages[target]["title"], "pllimit": "max"})
            # At most one matching link per source, so <=50 matches per batch.
            for page in raw.get("query", {}).get("pages", []):
                if page.get("pageid") in batch and any(
                        link.get("title") == self.pages[target]["title"]
                        for link in page.get("links", [])):
                    return [source, page["pageid"], target]
        return None


class LiveSemanticPolicy:
    """Same title+intro vectors and cosine/M1/M2 scores as offline navigation."""
    def __init__(self, graph, encoder_path, checkpoint=None):
        self.graph = graph
        self.encoder_path = encoder_path
        self.encoder = None
        self.vectors = {}
        self.model = None
        self.kind = "M0"
        if checkpoint:
            import torch
            from wikiclick_core.model.next_link.residual_ranker import ResidualRanker
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.kind = saved["identity"]["model"]
            if self.kind not in ("M1", "M2"):
                raise ValueError("Use an M1 or M2 navigation/scaling checkpoint")
            self.model = ResidualRanker(128)
            self.model.load_state_dict(saved["state_dict"], strict=True)
            self.model.eval().requires_grad_(False)

    def encode(self, nodes):
        import torch
        from torch.nn import functional as F
        missing = list(dict.fromkeys(n for n in nodes if n not in self.vectors))
        if not missing:
            return
        print(f"Encoding {len(missing)} articles for semantic ranking...", file=sys.stderr, flush=True)
        self.graph.ensure_texts(missing)
        if self.encoder is None:
            if not self.encoder_path.is_dir():
                raise ValueError(f"Local semantic encoder missing: {self.encoder_path}")
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(
                str(self.encoder_path), device="cpu", local_files_only=True,
                model_kwargs={"torch_dtype": torch.float32},
                config_kwargs={"vision_config": None, "audio_config": None})
            self.encoder.eval().requires_grad_(False)
        titles = [self.graph.pages[n]["title"] for n in missing]
        intros = [" ".join(self.graph.pages[n].get("extract", "").split()[:208]) for n in missing]
        texts = sorted(set(titles + [text for text in intros if text]))
        with torch.inference_mode():
            vectors = self.encoder.encode(
                texts, prompt_name="SentenceSimilarity", batch_size=16, truncate_dim=128,
                normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).float().cpu()
            lookup = dict(zip(texts, F.normalize(vectors, dim=1)))
            for node, title, intro in zip(missing, titles, intros):
                self.vectors[node] = F.normalize(
                    lookup[title] + (lookup[intro] if intro else torch.zeros(128)), dim=0)

    def score(self, current, candidates, target):
        import torch
        self.encode([*candidates, target])
        with torch.inference_mode():
            candidate = torch.stack([self.vectors[n] for n in candidates])
            goal = self.vectors[target].expand_as(candidate)
            cosine = (candidate * goal).sum(1)
            if self.model is None:
                return cosine
            correction, _ = self.model(candidate, goal)
            return cosine + correction if self.kind == "M2" else correction


def find_live_path(graph, source, target, policy, max_expansions=30, max_clicks=10,
                   short_route_check=False):
    from wikiclick_core.model.navigator.best_first import navigate
    if max_expansions < 1 or max_clicks < 1:
        raise ValueError("Search budgets must be positive")
    try:
        path = graph.short_route(source, target, max_clicks) if short_route_check else None
        if path is not None:
            result = {"path": path, "route_clicks": len(path) - 1, "status": "found",
                      "search_method": "short_route_check"}
        else:
            result = navigate(graph, source, target, policy, budgets=(max_expansions,),
                              max_clicks=max_clicks)[str(max_expansions)]
            result["search_method"] = "semantic_best_first"
    except RequestBudgetExceeded:
        result = {"path": None, "route_clicks": None, "status": "api_query_budget"}
    result.update({
        "route": [{"pageid": n, "title": graph.pages[n]["title"],
                   "url": graph.pages[n]["fullurl"]} for n in result["path"] or []],
        "api_queries": graph.requests,
        "fully_fetched_articles": len(graph.adjacency),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "optimality": "Candidate route; global shortest path has not been established",
        "scope": "Live English Wikipedia article links; redirects resolved; each link counts as one click",
        "limits": {"expansions": max_expansions, "clicks": max_clicks,
                   "api_queries": graph.max_requests},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Wikipedia URL or article title; prompts for a URL if omitted")
    parser.add_argument("--target", help="Wikipedia URL or article title; prompts for a URL if omitted")
    parser.add_argument("--max-expansions", type=int, default=30)
    parser.add_argument("--max-clicks", type=int, default=10)
    parser.add_argument("--max-api-queries", type=int, default=200,
                        help="Includes endpoint resolution and continuation; retries may add HTTP requests")
    parser.add_argument("--encoder", type=Path, default=ROOT / "data/cache/semantic_encoders/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e")
    parser.add_argument("--checkpoint", type=Path, help="Optional existing navigation/scaling M1 or M2 checkpoint; default: cosine M0")
    parser.add_argument("--output", type=Path, help="Save route, query counts and limits as JSON")
    args = parser.parse_args()
    if min(args.max_expansions, args.max_clicks, args.max_api_queries) < 1:
        parser.error("All budgets must be positive")
    try:
        if args.source is None or args.target is None:
            print("Enter valid English Wikipedia article links. Press Enter after each link.")
        if args.source is None:
            args.source = prompt_endpoint("Source")
        if args.target is None:
            args.target = prompt_endpoint("Target")
        # Validate both URLs before making any requests.
        article_title(args.source)
        article_title(args.target)
        import torch
        torch.set_num_threads(4)
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30,
                          follow_redirects=True) as http:
            graph = LiveGraph(http, args.max_api_queries)
            print("\nFetching Wikipedia pages and searching for a short route...", flush=True)
            source, target = graph.resolve(args.source), graph.resolve(args.target)
            print(f"Source: {graph.pages[source]['title']}\nTarget: {graph.pages[target]['title']}", flush=True)
            policy = LiveSemanticPolicy(graph, args.encoder, args.checkpoint)
            result = find_live_path(graph, source, target, policy,
                                    args.max_expansions, args.max_clicks, short_route_check=True)
            result["model"] = policy.kind
            result["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
            result["encoder"] = str(args.encoder)
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "\nSearch cancelled. You can also pass --source and --target.\n")
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as error:
        parser.exit(1, f"Cannot search live Wikipedia: {error}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["path"]:
        click_label = "click" if result["route_clicks"] == 1 else "clicks"
        print(f"\nCandidate shortest path ({result['route_clicks']} {click_label}):")
        print(" -> ".join(page["title"] for page in result["route"]))
        print()
        for step, page in enumerate(result["route"]):
            print(f"{step}. {page['title']}\n   {page['url']}")
    print(f"Status: {result['status']}; clicks: {result['route_clicks']}; API queries: {result['api_queries']}")
    print(result["optimality"] if result["path"] else "No route found within these limits; this does not establish that no route exists.")


if __name__ == "__main__":
    main()
