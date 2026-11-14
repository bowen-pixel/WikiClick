"""Fetch a bounded Wikipedia link dataset; this is a client, not a web server."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import wikipediaapi
import httpx

USER_AGENT = "WikiClick/0.1 (https://github.com/bowen-pixel/WikiClick)"
DEFAULT_TITLES = [
    "Graph theory",
    "Graph (discrete mathematics)",
    "Shortest path problem",
    "Dijkstra's algorithm",
    "Breadth-first search",
]


def query_api(http, params):
    """Read MediaWiki JSON, respecting temporary errors and Retry-After."""
    for attempt in range(4):
        try:
            response = http.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "format": "json", "formatversion": 2,
                "maxlag": 5, **params,
            })
            retry_after = response.headers.get("Retry-After", "")
            if response.status_code in (429, 503):
                delay = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            else:
                response.raise_for_status()
                result = response.json()
                if "error" not in result:
                    return result
                error = result["error"]
                if error.get("code") not in ("maxlag", "ratelimited"):
                    raise ValueError(f"MediaWiki API error: {error}")
                delay = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
        except httpx.TransportError:
            if attempt == 3:
                raise
            delay = 2 ** (attempt + 1)
        if attempt < 3:
            time.sleep(delay)
    raise RuntimeError("Wikipedia is temporarily unavailable after four attempts.")


def get_resolved_page_links(client, http, title):
    """Fetch one source with Wikipedia-API; resolve outgoing IDs via a generator.

    The package exposes outgoing titles but not batched redirect resolution.
    The public MediaWiki generator supplies canonical existing target IDs.
    Only return a record after all continuation pages have succeeded.
    """
    page = client.page(title)
    if not page.exists() or page.ns != wikipediaapi.Namespace.MAIN:
        raise ValueError(f"Not an existing article: {title}")
    targets = {}
    missing = set()
    params = {
        "generator": "links", "pageids": page.pageid,
        "gplnamespace": 0, "gpllimit": "max", "redirects": 1,
        "prop": "info", "inprop": "url",
    }
    while True:
        raw = query_api(http, params)
        for target in raw.get("query", {}).get("pages", []):
            if "missing" in target or "invalid" in target or "pageid" not in target:
                missing.add(target["title"])
            elif target["ns"] == 0:
                targets[target["pageid"]] = {
                    "pageid": target["pageid"], "title": target["title"],
                    "url": target["fullurl"],
                }
        if "continue" not in raw:
            break
        params.update(raw["continue"])
    return {
        "requested_title": title, "title": page.title,
        "pageid": page.pageid, "url": page.fullurl,
        "links": sorted(item["title"] for item in targets.values()),
        "link_targets": sorted(targets.values(), key=lambda item: item["pageid"]),
        "missing_link_titles": sorted(missing), "links_complete": True,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def create_client():
    """Identify our project and let the library handle pagination and retries."""
    return wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
        timeout=20,
        max_retries=2,
        extra_api_params={"maxlag": 5},
    )


def get_page_links(client, title):
    """Return plain data for one existing article and its article-link titles.

    Calling exists() loads metadata and resolves the requested source title.
    Link objects are lazy stubs: do not fetch metadata for every outgoing link.
    Failures propagate so an API outage is never recorded as an empty link set.
    """
    page = client.page(title)
    if not page.exists():
        raise ValueError(f"Article does not exist: {title}")
    if page.ns != wikipediaapi.Namespace.MAIN:
        raise ValueError(f"Only article pages are supported: {title}")
    links = client.links(page, plnamespace=0, pllimit="max")
    return {
        "requested_title": title,
        "title": page.title,
        "pageid": page.pageid,
        "url": page.fullurl,
        "links": sorted(
            name for name, linked in links.items()
            if linked.ns == wikipediaapi.Namespace.MAIN
        ),
    }


def collect_domain(client, titles):
    """Fetch only explicitly selected sources, sequentially; never crawl links."""
    titles = list(dict.fromkeys(titles))
    if not 1 <= len(titles) <= 20:
        raise ValueError("Select between 1 and 20 article titles for this prototype.")
    pages = []
    for title in titles:
        print(f"Fetching {title}...", file=sys.stderr, flush=True)
        pages.append(get_page_links(client, title))
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": "https://en.wikipedia.org/w/api.php",
        "language": "en",
        "user_agent": USER_AGENT,
        "scope": "Only selected articles; outgoing targets are title stubs.",
        "pages": pages,
    }


def prompt_domain():
    """Collect endpoints and optional intermediate articles, not a full crawl."""
    print("Choose the articles to include in your small graph.")
    print("Only links between included articles will be used for searching.")
    titles = []
    for label in ("source", "target"):
        title = ""
        while not title:
            title = input(f"Enter {label}: ").strip()
        titles.append(title)
    print("Add possible intermediate articles, one per line; press Enter to finish.")
    while len(set(titles)) < 20:
        title = input("Additional article: ").strip()
        if not title:
            break
        titles.append(title)
    return list(dict.fromkeys(titles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("titles", nargs="*", help="Up to 20 articles; defaults to five graph topics")
    parser.add_argument("--output", type=Path, default=Path("data/graphs/small_domain.json"))
    parser.add_argument("--interactive", action="store_true", help="Prompt for source, target, and optional intermediate articles")
    args = parser.parse_args()
    if args.interactive and args.titles:
        parser.error("Use either --interactive or positional article titles, not both.")
    try:
        titles = prompt_domain() if args.interactive else args.titles or DEFAULT_TITLES
        data = collect_domain(create_client(), titles)
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "\nInput cancelled; no dataset written.\n")
    except (ValueError, wikipediaapi.WikiHttpError,
            wikipediaapi.WikiConnectionError, wikipediaapi.WikiHttpTimeoutError,
            wikipediaapi.WikiInvalidJsonError) as error:
        parser.exit(1, f"Collection failed; no dataset written: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(data['pages'])} article records to {args.output}")


if __name__ == "__main__":
    main()
