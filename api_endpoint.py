"""Fetch a bounded Wikipedia link dataset; this is a client, not a web server."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import wikipediaapi

USER_AGENT = "WikiClick/0.1 (https://github.com/bowen-pixel/WikiClick)"
DEFAULT_TITLES = [
    "Graph theory",
    "Graph (discrete mathematics)",
    "Shortest path problem",
    "Dijkstra's algorithm",
    "Breadth-first search",
]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("titles", nargs="*", help="Up to 20 articles; defaults to five graph topics")
    parser.add_argument("--output", type=Path, default=Path("data/graphs/small_domain.json"))
    args = parser.parse_args()
    try:
        data = collect_domain(create_client(), args.titles or DEFAULT_TITLES)
    except (ValueError, wikipediaapi.WikiHttpError,
            wikipediaapi.WikiConnectionError, wikipediaapi.WikiHttpTimeoutError,
            wikipediaapi.WikiInvalidJsonError) as error:
        parser.exit(1, f"Collection failed; no dataset written: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(data['pages'])} article records to {args.output}")


if __name__ == "__main__":
    main()
