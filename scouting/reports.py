"""Scouting report search (RAG): "which Mets prospects have a plus changeup
but command concerns?"

Scouts write in their own language -- "60-grade slider", "power over hit",
"effort in the delivery" -- so keyword search misses most matches. Text
embeddings match on MEANING, which is exactly what RAG is for.

Where the reports come from: you collect them (copy from MLB Pipeline,
FanGraphs, Baseball America, your own notes) into data/scouting/reports.csv.
Keep that file local -- published reports are copyrighted, so don't commit
it to a public repo (it's in .gitignore).

Columns: player, team, position, source, date, url, text
Then build:   python -m scouting.reports
"""

from __future__ import annotations

import csv
import re
import sys

from search.env import load_env
from search.store import DATA, VectorStore, text_embedder

CSV_PATH = DATA / "scouting" / "reports.csv"
STORE = DATA / "scouting_index"
BUILD_HINT = "add reports to data/scouting/reports.csv, then: python -m scouting.reports"
FIELDS = ["player", "team", "position", "source", "date", "url", "text"]
MAX_CHARS = 1200  # longer reports get split into paragraph-sized chunks


def load_reports(path=None) -> list[dict]:
    with open(path or CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("text") or "").strip()]
    return [r for r in rows if not r["player"].upper().startswith("EXAMPLE")]


def chunk_report(report: dict) -> list[dict]:
    """Short reports stay whole. Long ones are split on paragraph breaks,
    each chunk keeping the player/team metadata AND starting with the
    player's name -- so a chunk about "his changeup" still knows whose."""
    text = report["text"].strip()
    if len(text) <= MAX_CHARS:
        parts = [text]
    else:
        parts, cur = [], ""
        for para in re.split(r"\n\s*\n", text):
            if cur and len(cur) + len(para) > MAX_CHARS:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{para}".strip()
        parts.append(cur)
    meta = {k: report.get(k, "") for k in FIELDS if k != "text"}
    return [{**meta, "part": i + 1, "text": p,
             "embed_text": f"{report['player']} ({report.get('position', '')}, {report.get('team', '')}): {p}"}
            for i, p in enumerate(parts)]


def build(embed=None) -> VectorStore:
    embed = embed or text_embedder()
    chunks = [c for r in load_reports() for c in chunk_report(r)]
    if not chunks:
        raise ValueError(f"No reports found in {CSV_PATH}. {BUILD_HINT}")
    store = VectorStore(STORE, chunks, embed([c["embed_text"] for c in chunks]))
    store.save()
    return store


def search(store: VectorStore, query: str, embed, team: str | None = None,
           position: str | None = None, k: int = 5) -> list[dict]:
    def where(it: dict) -> bool:
        return ((not team or team.lower() in it["team"].lower())
                and (not position or position.lower() in it["position"].lower()))

    hits = store.search(embed([query])[0], k=k, where=where)
    return [{"score": s, **{k2: v for k2, v in it.items() if k2 != "embed_text"}} for s, it in hits]


def main() -> None:
    load_env()
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerow(["EXAMPLE Player (delete this row)", "Mets", "RHP", "your notes", "2026-11-16", "",
                        "Replace this row with a real report you copied in."])
        sys.exit(f"Created {CSV_PATH} -- add reports, then run this again.")
    store = build()
    print(f"{len(load_reports())} reports -> {len(store.items)} chunks -> {STORE}")


if __name__ == "__main__":
    main()
