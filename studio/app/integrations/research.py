"""Free research sources: Reddit, Hacker News (Algolia) and Product Hunt.

* Reddit: app-only OAuth when REDDIT_CLIENT_ID/SECRET are set (reliable from cloud IPs),
  else the public JSON endpoints (often rate limited from datacenters).
* Hacker News: hn.algolia.com, no key, generous limits.
* Product Hunt: GraphQL v2 with a free developer token (optional).

X/Twitter and Google Trends are left out on purpose: X's API is paid, and Google Trends has no
official API (scrapers break and get blocked). The Head of Product works fine without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.config import Settings, secret

log = logging.getLogger(__name__)

QUERIES = (
    "automate", "n8n workflow", "is there a tool", "zapier alternative", "no-show", "client onboarding",
    "invoice reminder", "lead generation automation", "manually every week",
)


@dataclass(frozen=True)
class RawPost:
    source: str
    external_id: str
    title: str
    url: str
    text: str
    engagement: int


class RedditSource:
    name = "reddit"

    def __init__(self, s: Settings, client: httpx.AsyncClient) -> None:
        self.s, self.client = s, client
        self._token = ""

    async def _auth(self) -> dict[str, str]:
        headers = {"User-Agent": self.s.reddit_user_agent}
        if self.s.reddit_client_id and self.s.reddit_client_secret:
            if not self._token:
                r = await self.client.post("https://www.reddit.com/api/v1/access_token",
                                           data={"grant_type": "client_credentials"}, headers=headers,
                                           auth=(self.s.reddit_client_id, secret(self.s.reddit_client_secret)))
                r.raise_for_status()
                self._token = r.json()["access_token"]
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def fetch(self, limit_per_query: int = 15) -> list[RawPost]:
        headers = await self._auth()
        base = "https://oauth.reddit.com" if "Authorization" in headers else "https://www.reddit.com"
        out: list[RawPost] = []
        subs = "+".join(self.s.subreddits)
        for q in QUERIES[:5]:
            r = await self.client.get(f"{base}/r/{subs}/search.json", headers=headers,
                                      params={"q": q, "restrict_sr": 1, "sort": "new", "t": "month", "limit": limit_per_query})
            if r.status_code != 200:
                log.info("reddit %s -> HTTP %s", q, r.status_code)
                continue
            for child in r.json().get("data", {}).get("children", []):
                d = child.get("data", {})
                out.append(RawPost("reddit", d.get("id", ""), d.get("title", ""),
                                   f"https://www.reddit.com{d.get('permalink', '')}", d.get("selftext", "")[:3000],
                                   int(d.get("score", 0)) + 2 * int(d.get("num_comments", 0))))
        return out


class HackerNewsSource:
    name = "hackernews"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, limit_per_query: int = 20) -> list[RawPost]:
        out: list[RawPost] = []
        for q in QUERIES:
            r = await self.client.get("https://hn.algolia.com/api/v1/search_by_date",
                                      params={"query": q, "tags": "(story,comment,ask_hn)", "hitsPerPage": limit_per_query})
            if r.status_code != 200:
                continue
            for h in r.json().get("hits", []):
                title = h.get("title") or h.get("story_title") or ""
                text = h.get("comment_text") or h.get("story_text") or ""
                out.append(RawPost("hackernews", str(h.get("objectID")), title[:500],
                                   f"https://news.ycombinator.com/item?id={h.get('objectID')}", text[:3000],
                                   int(h.get("points") or 0) + int(h.get("num_comments") or 0)))
        return out


class ProductHuntSource:
    name = "producthunt"
    QUERY = """query($topic:String!){posts(topic:$topic, first:20, order:NEWEST){edges{node{id name tagline url votesCount commentsCount description}}}}"""

    def __init__(self, s: Settings, client: httpx.AsyncClient) -> None:
        self.s, self.client = s, client

    async def fetch(self) -> list[RawPost]:
        out: list[RawPost] = []
        for topic in ("productivity", "marketing-automation", "no-code", "sales"):
            r = await self.client.post("https://api.producthunt.com/v2/api/graphql",
                                       json={"query": self.QUERY, "variables": {"topic": topic}},
                                       headers={"Authorization": f"Bearer {secret(self.s.producthunt_token)}"})
            if r.status_code != 200:
                continue
            for e in r.json().get("data", {}).get("posts", {}).get("edges", []):
                n = e["node"]
                out.append(RawPost("producthunt", str(n["id"]), f"{n['name']}: {n['tagline']}", n["url"],
                                   (n.get("description") or "")[:2000], int(n["votesCount"]) + int(n["commentsCount"])))
        return out


async def mine(s: Settings, client: httpx.AsyncClient) -> tuple[list[RawPost], dict[str, str]]:
    """All configured sources. Returns posts and a per-source status (ok / error text)."""
    sources: list[RedditSource | HackerNewsSource | ProductHuntSource] = [RedditSource(s, client), HackerNewsSource(client)]
    if s.producthunt_token:
        sources.append(ProductHuntSource(s, client))
    posts: list[RawPost] = []
    status: dict[str, str] = {}
    for src in sources:
        try:
            got = await src.fetch()
            posts += got
            status[src.name] = f"ok ({len(got)})"
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            status[src.name] = f"error: {type(exc).__name__}"
            log.warning("source %s failed: %s", src.name, exc)
    return posts, status
