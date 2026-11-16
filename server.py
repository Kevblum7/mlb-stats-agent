"""MLB stats MCP server.

An MCP server exposes TOOLS that any MCP client (the OpenAI Agents SDK,
Claude Desktop, Cursor...) can discover and call. The client never sees this
code -- only each tool's name, its type-hinted parameters and its docstring.
So the docstring IS the tool's instructions to the model: write it for an
LLM reader.

Run it standalone to check it starts:   python server.py
(It will sit waiting for a client on stdin/stdout -- Ctrl+C to stop.)

Note: this uses the MCP Python SDK v2, where the v1 class `FastMCP` was
renamed `MCPServer`. Tutorials written for v1 use `FastMCP`; same idea.
"""

from __future__ import annotations

import contextlib
import sys
from datetime import date

from mcp.server.mcpserver import MCPServer
import pybaseball
import requests

pybaseball.cache.enable()  # pybaseball scrapes slow sites; cache to disk

def quiet():
    """Send anything a library PRINTS to stderr instead of stdout.

    Real bug we hit building this: with the stdio transport, stdout IS the
    JSON-RPC channel to the client. pybaseball prints "Gathering player
    lookup table..." to stdout, the client tries to parse that as JSON and
    the connection dies. Wrap every pybaseball call in `with quiet():`.
    """
    return contextlib.redirect_stdout(sys.stderr)


mcp = MCPServer(
    name="mlb-stats",
    instructions="Tools for looking up MLB players and their statistics.",
)


# --------------------------------------------------------------------------
# Worked example -- read this one, then write the next two yourself.
# --------------------------------------------------------------------------

@mcp.tool()
def lookup_player(first_name: str, last_name: str) -> list[dict]:
    """Find MLB players by name and return their IDs.

    Call this FIRST whenever the user names a player: season_stats and
    pitch_mix both need the `mlbam_id` it returns.
    Several players can share a name -- if more than one comes back, use
    the years played to pick the right one, or ask the user.
    """
    with quiet():
        df = pybaseball.playerid_lookup(last_name.strip(), first_name.strip(), fuzzy=True)
    if df.empty:
        return []
    return [
        {
            "name": f"{r.name_first} {r.name_last}".title(),
            "mlbam_id": int(r.key_mlbam),
            "fangraphs_id": int(r.key_fangraphs) if r.key_fangraphs > 0 else None,
            "first_year": int(r.mlb_played_first) if r.mlb_played_first == r.mlb_played_first else None,
            "last_year": int(r.mlb_played_last) if r.mlb_played_last == r.mlb_played_last else None,
        }
        for r in df.head(5).itertuples()
    ]
    # Design notes:
    # - Returns plain JSON-friendly types (no DataFrames): the result is
    #   serialized and shown to the model.
    # - Caps results at 5: tool output costs tokens and attention.
    # - An empty list is a valid answer, not an error. The model can say
    #   "I couldn't find that player" instead of crashing.


# --------------------------------------------------------------------------
# YOUR TURN. Write these two. The docstrings are drafts -- improve them.
# --------------------------------------------------------------------------

@mcp.tool()
def season_stats(mlbam_id: int, year: int, kind: str = "batting") -> dict:
    """Regular-season totals for one player and year, from MLB's Stats API.

    Call lookup_player first and pass its mlbam_id. kind is "batting" or
    "pitching" (use "pitching" for pitchers).

    Batting returns G, PA, HR, AVG, OBP, SLG, OPS.
    Pitching returns G, IP, ERA, WHIP, K, BB, K%, BB%.
    WAR, wRC+ and FIP are not available from this source; say so if asked.

    Returns {"error": "..."} if the player has no stats of that kind that
    year (wrong id, wrong kind, or didn't play). Do not invent stats.
    """
    if kind not in ("batting", "pitching"):
        return {"error": 'kind must be "batting" or "pitching".'}
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}/stats",
            params={"stats": "season", "season": year, "group": "hitting" if kind == "batting" else "pitching"},
            timeout=20,
        )
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"] if resp.json().get("stats") else []
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return {"error": "MLB Stats API request failed. Try again once; if it fails again, say the stats are unavailable."}

    # A traded player has one split per team plus a season total with no team.
    totals = [s for s in splits if "team" not in s] or splits[-1:]
    if not totals:
        return {
            "error": (
                f"No {kind} stats for mlbam_id {mlbam_id} in {year}. "
                "Check the id from lookup_player, the kind, and that they played that season."
            )
        }
    s = totals[0]["stat"]

    if kind == "batting":
        return {
            "G": s.get("gamesPlayed"),
            "PA": s.get("plateAppearances"),
            "HR": s.get("homeRuns"),
            "AVG": s.get("avg"),
            "OBP": s.get("obp"),
            "SLG": s.get("slg"),
            "OPS": s.get("ops"),
        }
    bf = s.get("battersFaced") or 0
    return {
        "G": s.get("gamesPlayed"),
        "IP": s.get("inningsPitched"),
        "ERA": s.get("era"),
        "WHIP": s.get("whip"),
        "K": s.get("strikeOuts"),
        "BB": s.get("baseOnBalls"),
        "K%": round(100 * s.get("strikeOuts", 0) / bf, 1) if bf else None,
        "BB%": round(100 * s.get("baseOnBalls", 0) / bf, 1) if bf else None,
    }


@mcp.tool()
def pitch_mix(mlbam_id: int, start_date: str, end_date: str) -> dict:
    """How often a pitcher threw each pitch type, with average velocity (mph).

    Call lookup_player first and pass its mlbam_id (not fangraphs_id).
    Dates are YYYY-MM-DD, at most 366 days apart. For a full season use
    roughly YYYY-03-15 to YYYY-11-05.

    Returns {"total_pitches": N, "pitches": [{"pitch", "count",
    "usage_pct", "avg_velo"}, ...]}, most used first. Returns
    {"error": "..."} for bad dates or when no pitches were found (wrong id,
    not a pitcher, or no games in that range).
    """
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return {"error": "Dates must be YYYY-MM-DD, e.g. 2025-04-01."}
    if end < start:
        return {"error": "end_date is before start_date."}
    if (end - start).days > 366:
        return {"error": "Range is over one season. Call once per season and compare."}

    with quiet():
        df = pybaseball.statcast_pitcher(start_date, end_date, mlbam_id)
    df = df.dropna(subset=["pitch_name"]) if "pitch_name" in df.columns else df.iloc[0:0]
    if df.empty:
        return {
            "error": (
                f"No pitches for mlbam_id {mlbam_id} from {start_date} to {end_date}. "
                "Check the id is the pitcher's mlbam_id from lookup_player and that "
                "they pitched in that range."
            )
        }

    total = len(df)
    mix = df.groupby("pitch_name").agg(count=("pitch_name", "size"), avg_velo=("release_speed", "mean"))
    mix = mix.sort_values("count", ascending=False)
    return {
        "total_pitches": total,
        "pitches": [
            {
                "pitch": name,
                "count": int(r["count"]),
                "usage_pct": round(100 * float(r["count"]) / total, 1),
                "avg_velo": None if r["avg_velo"] != r["avg_velo"] else round(float(r["avg_velo"]), 1),
            }
            for name, r in mix.iterrows()
        ],
    }


from extra_tools import register  # noqa: E402

register(mcp)


if __name__ == "__main__":
    mcp.run()  # stdio transport: the client launches this file as a subprocess
