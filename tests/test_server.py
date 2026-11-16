"""Checks the MCP plumbing without an OpenAI key: start server.py the same
way the agent does, and confirm the client sees our tools and schemas."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as app  # noqa: E402


def test_client_discovers_tools():
    async def go():
        async with app.stats_server() as server:
            return {t.name: t for t in await server.list_tools()}

    tools = asyncio.run(go())
    assert set(tools) == {"lookup_player", "season_stats", "pitch_mix",
                          "similar_pitches", "search_scouting", "search_news"}
    # The model only sees name + schema + description -- check they're there.
    lookup = tools["lookup_player"]
    assert set(lookup.input_schema["required"]) == {"first_name", "last_name"}
    assert "Call this FIRST" in lookup.description
