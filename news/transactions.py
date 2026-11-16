"""News & transaction memory: "what has happened with Alonso's wrist?",
"what bullpen moves did the Mets make this month?"

Source: the free MLB Stats API transactions feed -- official, factual
(IL placements, activations, trades, signings, options, DFAs). No scraping.

Two things make this different from the other search tools:
  * it goes STALE, so you refresh it:   python -m news.transactions --days 120
  * recent beats old. Ranking blends meaning-similarity with recency
    (RECENCY_WEIGHT, HALF_LIFE_DAYS) so last week's IL move outranks a
    near-identical one from April.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

from search.env import load_env
from search.store import DATA, VectorStore, text_embedder

STORE = DATA / "news_index"
BUILD_HINT = "python -m news.transactions --days 120"
API = "https://statsapi.mlb.com/api/v1/transactions"
RECENCY_WEIGHT = 0.25   # 0 = pure meaning match, 1 = newest first regardless of meaning
HALF_LIFE_DAYS = 30     # a transaction's recency bonus halves every 30 days


def fetch(start: date, end: date, team_id: int | None = None) -> list[dict]:
    params = {"startDate": start.isoformat(), "endDate": end.isoformat()}
    if team_id:
        params["teamId"] = team_id
    with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=60) as r:
        return json.load(r).get("transactions", [])


def to_items(raw: list[dict]) -> list[dict]:
    """Keep the fields we search and cite; drop rows with no description."""
    items, seen = [], set()
    for t in raw:
        desc = (t.get("description") or "").strip()
        if not desc or t.get("id") in seen:
            continue
        seen.add(t.get("id"))
        items.append({
            "id": t.get("id"),
            "date": (t.get("effectiveDate") or t.get("date") or "")[:10],
            "player": (t.get("person") or {}).get("fullName", ""),
            "player_id": (t.get("person") or {}).get("id"),
            "team": (t.get("toTeam") or t.get("fromTeam") or {}).get("name", ""),
            "from_team": (t.get("fromTeam") or {}).get("name", ""),
            "type": t.get("typeDesc", ""),
            "description": desc,
        })
    return items


def build(raw: list[dict], embed=None) -> VectorStore:
    embed = embed or text_embedder()
    items = to_items(raw)
    if not items:
        raise ValueError("No transactions returned for that range.")
    store = VectorStore(STORE, items, embed([f"{i['date']} {i['type']}: {i['description']}" for i in items]))
    store.save()
    return store


def recency(d: str, today: date) -> float:
    try:
        age = (today - date.fromisoformat(d)).days
    except ValueError:
        return 0.0
    return 0.5 ** (max(age, 0) / HALF_LIFE_DAYS)


def search(store: VectorStore, query: str, embed, team: str | None = None,
           player: str | None = None, since_days: int | None = None, k: int = 8,
           today: date | None = None) -> dict:
    today = today or date.today()
    cutoff = (today - timedelta(days=since_days)).isoformat() if since_days else ""

    def where(it: dict) -> bool:
        return ((not team or team.lower() in f"{it['team']} {it['from_team']}".lower())
                and (not player or player.lower() in it["player"].lower())
                and it["date"] >= cutoff)

    scored = store.scored(embed([query])[0], where=where)
    ranked = sorted((((1 - RECENCY_WEIGHT) * s + RECENCY_WEIGHT * recency(it["date"], today), s, it)
                     for s, it in scored), key=lambda t: t[0], reverse=True)[:k]
    newest = max((it["date"] for it in store.items), default=None)
    return {
        "data_through": newest,
        "results": [{"score": round(c, 3), "similarity": round(s, 3), **it} for c, s, it in ranked],
    }


def main() -> None:
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--team-id", type=int, help="e.g. 121 = Mets. Omit for all teams.")
    a = p.parse_args()
    end = date.today()
    raw: list[dict] = []
    start = end - timedelta(days=a.days)
    while start < end:  # fetch in 30-day windows to keep each response small
        stop = min(start + timedelta(days=30), end)
        raw += fetch(start, stop, a.team_id)
        start = stop + timedelta(days=1)
    store = build(raw)
    print(f"{len(raw)} transactions -> {len(store.items)} indexed (through {max(i['date'] for i in store.items)}) -> {STORE}")


if __name__ == "__main__":
    main()
