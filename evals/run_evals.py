"""Evals: does the agent get baseball questions RIGHT?

Each case = a question + checks on (a) which tools the agent called and
(b) what its answer says. Run:

    python -m evals.run_evals
    python -m evals.run_evals --runs 3     # agents vary; check consistency

Cases mix:
  - easy lookups (one player, one season)
  - multi-step (two players compared, two seasons, pitch mix)
  - traps: a misspelled name, a name shared by two players, a season not
    played yet, a stat the tools don't have, a hitter's "pitch mix"
Number checks use true values hard-coded with a tolerance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Agent, Runner  # noqa: E402
from agents.items import ToolCallItem  # noqa: E402

import agent as app  # noqa: E402


@dataclass
class Run:
    answer: str
    tools_called: list[str]
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)


@dataclass
class Case:
    name: str
    question: str
    checks: dict[str, Callable[[Run], bool]] = field(default_factory=dict)


def number_near(text: str, target: float, tol: float) -> bool:
    """True if any number in `text` is within `tol` of `target`."""
    return any(abs(float(n) - target) <= tol for n in re.findall(r"-?\d+\.?\d*", text))


def parse_args(raw) -> dict:
    try:
        return json.loads(getattr(raw, "arguments", "") or "{}")
    except json.JSONDecodeError:
        return {}


def called_with(r: Run, tool: str, **expected) -> bool:
    """True if `tool` was called at least once with all of `expected` as arguments."""
    return any(
        name == tool and all(args.get(k) == v for k, v in expected.items())
        for name, args in r.tool_calls
    )


def says(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I) is not None


ALONSO, DEVERS, SKENES, WILL_SMITH_C = 624413, 646240, 694973, 669257
ASKS_WHICH = r"which one|which (player|will smith)|two (\w+ )?players|specify|clarify|did you mean|do you mean"


CASES = [
    # Example 1: checks the PROCESS (right tool, called first).
    Case(
        "resolves_name_first",
        "How many home runs did Pete Alonso hit in 2024?",
        {
            "called lookup_player": lambda r: "lookup_player" in r.tools_called,
            "lookup came first": lambda r: r.tools_called[:1] == ["lookup_player"],
            "called season_stats": lambda r: "season_stats" in r.tools_called,
            "asked for Alonso's 2024": lambda r: called_with(r, "season_stats", mlbam_id=ALONSO, year=2024),
            "correct HR total": lambda r: number_near(r.answer, 34, 0),
        },
    ),
    # Example 2: a trap -- the agent should NOT invent a number.
    Case(
        "no_such_season",
        "What was Juan Soto's ERA in 2024?",
        {
            "doesn't invent an ERA": lambda r: not re.search(r"\b\d\.\d\d\b", r.answer),
        },
    ),

    # --- Easy lookups ---
    Case(
        "pitcher_season",
        "What was Paul Skenes' ERA in 2025?",
        {
            "lookup came first": lambda r: r.tools_called[:1] == ["lookup_player"],
            "asked for pitching stats": lambda r: called_with(
                r, "season_stats", mlbam_id=SKENES, year=2025, kind="pitching"),
            "correct ERA (1.97)": lambda r: number_near(r.answer, 1.97, 0.005),
        },
    ),
    Case(
        "traded_player_full_season",
        "How many home runs did Rafael Devers hit in 2025?",
        {
            "asked for Devers's 2025": lambda r: called_with(r, "season_stats", mlbam_id=DEVERS, year=2025),
            # Traded mid-season: 35 is the combined total, not one team's share.
            "full-season total (35)": lambda r: number_near(r.answer, 35, 0),
        },
    ),

    # --- Multi-step ---
    Case(
        "compare_two_players",
        "Who hit more home runs in 2025, Pete Alonso or Rafael Devers?",
        {
            "looked up both players": lambda r: r.tools_called.count("lookup_player") >= 2,
            "fetched Alonso's 2025": lambda r: called_with(r, "season_stats", mlbam_id=ALONSO, year=2025),
            "fetched Devers's 2025": lambda r: called_with(r, "season_stats", mlbam_id=DEVERS, year=2025),
            "Alonso paired with 38": lambda r: says(r.answer, r"Alonso[^.\n]{0,60}\b38\b"),
            "Devers paired with 35": lambda r: says(r.answer, r"Devers[^.\n]{0,60}\b35\b"),
            "names Alonso as the leader": lambda r: says(
                r.answer,
                r"Alonso[^.\n]{0,80}\b(more|most|led|ahead|edge|outhit|out-homered)\b"
                r"|Devers[^.\n]{0,80}\b(fewer|less|behind|trailed)\b",
            ),
        },
    ),
    Case(
        "compare_two_seasons",
        "Compare Pete Alonso's home runs in 2024 and 2025.",
        {
            "fetched 2024": lambda r: called_with(r, "season_stats", mlbam_id=ALONSO, year=2024),
            "fetched 2025": lambda r: called_with(r, "season_stats", mlbam_id=ALONSO, year=2025),
            "2024 HR (34)": lambda r: number_near(r.answer, 34, 0),
            "2025 HR (38)": lambda r: number_near(r.answer, 38, 0),
        },
    ),
    Case(
        "pitch_mix_top_pitch",
        "What pitch did Paul Skenes throw most in April 2025?",
        {
            "lookup came first": lambda r: r.tools_called[:1] == ["lookup_player"],
            "pitch_mix for Skenes in April 2025": lambda r: any(
                name == "pitch_mix" and args.get("mlbam_id") == SKENES
                and str(args.get("start_date", "")).startswith("2025-04")
                for name, args in r.tool_calls
            ),
            "names the four-seam fastball": lambda r: says(r.answer, r"4-seam|four-seam|fastball"),
        },
    ),
    Case(
        "pitch_mix_velocity",
        "How hard did Paul Skenes throw his four-seam fastball in April 2025?",
        {
            "called pitch_mix": lambda r: "pitch_mix" in r.tools_called,
            "correct velocity (98.0 mph)": lambda r: number_near(r.answer, 98.0, 0.3),
        },
    ),

    # --- Traps ---
    Case(
        "misspelled_name",
        "How many home runs did Pete Alonzo hit in 2025?",
        {
            "resolved to Alonso": lambda r: called_with(r, "season_stats", mlbam_id=ALONSO, year=2025),
            "correct HR total (38)": lambda r: number_near(r.answer, 38, 0),
        },
    ),
    Case(
        # Two Will Smiths: a catcher (20 HR in 2024) and a pitcher (no batting stats).
        "shared_name",
        "How many home runs did Will Smith hit in 2024?",
        {
            "picks the catcher or asks which one": lambda r: called_with(
                r, "season_stats", mlbam_id=WILL_SMITH_C, year=2024) or says(r.answer, ASKS_WHICH),
            "doesn't give a wrong total": lambda r: number_near(r.answer, 20, 0) or says(r.answer, ASKS_WHICH),
        },
    ),
    Case(
        "unavailable_stat",
        "What was Pete Alonso's WAR in 2025?",
        {
            "says WAR isn't available": lambda r: says(
                r.answer, r"not available|unavailable|isn't available|don't have|do not have|can't|cannot|"
                          r"couldn't|could not|no WAR|not provide|doesn't (include|provide)"),
            "doesn't invent a WAR": lambda r: not says(r.answer, r"WAR[^.\n]{0,20}\b\d\.\d\b"),
        },
    ),
    Case(
        "season_not_played_yet",
        "How many home runs did Pete Alonso hit in 2027?",
        {
            "doesn't invent a total": lambda r: not says(r.answer, r"\b\d{1,2} (home runs|HR)"),
        },
    ),
    Case(
        "hitter_has_no_pitch_mix",
        "What was Pete Alonso's pitch mix in April 2025?",
        {
            "doesn't invent pitch usage": lambda r: not re.search(r"\d+(\.\d)?\s?%", r.answer),
        },
    ),
]


async def run_case(case: Case, server) -> Run:
    a = Agent(name="MLB Stats Assistant", instructions=app.INSTRUCTIONS,
              model=app.os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), mcp_servers=[server])
    result = await Runner.run(a, case.question, max_turns=10)
    calls = [i.raw_item for i in result.new_items if isinstance(i, ToolCallItem)]
    tools = [getattr(c, "name", "?") for c in calls]
    tool_calls = [(name, parse_args(c)) for name, c in zip(tools, calls)]
    return Run(answer=result.final_output or "", tools_called=tools, tool_calls=tool_calls)


async def main() -> int:
    app.load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--only")
    args = p.parse_args()

    # Imported here: addon_cases imports Case from this module.
    from evals.addon_cases import ADDON_CASES

    passed = total = 0
    async with app.stats_server() as server:
        for case in CASES + ADDON_CASES:
            if args.only and case.name != args.only:
                continue
            for k in range(args.runs):
                run = await run_case(case, server)
                print(f"\n=== {case.name} (run {k + 1}) tools={run.tools_called}")
                failed = False
                for label, check in case.checks.items():
                    ok = bool(check(run))
                    passed += ok
                    total += 1
                    failed |= not ok
                    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
                if failed:
                    print("  calls: ", run.tool_calls)
                    print("  answer:", run.answer.replace("\n", "\n          "))
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
