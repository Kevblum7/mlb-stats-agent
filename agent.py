"""The agent: OpenAI Agents SDK + our MCP server.

    python agent.py "Compare Pete Alonso's 2024 and 2025 seasons"
    python agent.py            # interactive chat

The SDK launches server.py as a subprocess, asks it for its tool list, and
hands those tools to the model. The model decides which to call; the SDK
runs the loop (call model -> run tools -> feed results back -> repeat).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from agents import Agent, Runner, SQLiteSession
from agents.mcp import MCPServerStdio

ROOT = Path(__file__).resolve().parent

INSTRUCTIONS = """
You are a baseball research assistant for a front office.
- Use tools for every number. Never state a stat from memory.
- Always resolve player names with lookup_player first.
- Cite the season or date range for each number you give.
- If a tool returns an error, read it and try a sensible next step once;
  if it still fails, say what you couldn't get.
- No data is not zero. If a season hasn't been played or a tool has no
  stats for it, say there's no data; never report 0 for it.
- For "who throws a pitch like X" questions, use similar_pitches (after lookup_player).
- For scouting/skills questions ("plus changeup", "raw power"), use search_scouting and cite the source.
- For injuries, trades and roster moves, use search_news and give each transaction's date.
- Combine tools when a question needs both words and numbers.
- Keep answers short: a sentence of takeaway, then the key numbers.
""".strip()


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def stats_server() -> MCPServerStdio:
    return MCPServerStdio(
        name="mlb-stats",
        params={"command": sys.executable, "args": [str(ROOT / "server.py")]},
        client_session_timeout_seconds=120,  # Statcast pulls can be slow
        cache_tools_list=True,
    )


async def ask(question: str, server: MCPServerStdio, session: SQLiteSession | None = None) -> str:
    agent = Agent(
        name="MLB Stats Assistant",
        instructions=INSTRUCTIONS,
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        mcp_servers=[server],
    )
    result = await Runner.run(agent, question, max_turns=10, session=session)
    return result.final_output


async def main() -> None:
    load_env()
    async with stats_server() as server:
        if len(sys.argv) > 1:
            print(await ask(" ".join(sys.argv[1:]), server))
            return
        session = SQLiteSession("chat")
        print("Ask about MLB players (blank line to quit).")
        while q := input("\n> ").strip():
            print(await ask(q, server, session))


if __name__ == "__main__":
    asyncio.run(main())
