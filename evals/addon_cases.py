"""Agent eval cases for the three search tools. Process checks only (did it
pick the right tool, and did it cite dates/sources?), so they work with
whatever data you've indexed.

Use them by adding two lines to evals/run_evals.py, right after CASES = [...]:

    from evals.addon_cases import ADDON_CASES
    CASES += ADDON_CASES
"""

from evals.run_evals import Case

ADDON_CASES = [
    Case(
        "comps_uses_similar_pitches",
        "Whose slider looks most like Edwin Diaz's?",
        {
            "looked him up first": lambda r: r.tools_called[:1] == ["lookup_player"],
            "called similar_pitches": lambda r: "similar_pitches" in r.tools_called,
        },
    ),
    Case(
        "scouting_uses_rag_not_stats",
        "Which Mets pitching prospects have a good changeup but shaky command?",
        {
            "called search_scouting": lambda r: "search_scouting" in r.tools_called,
            "didn't use stats tools for a scouting question": lambda r: "season_stats" not in r.tools_called,
        },
    ),
    Case(
        "news_cites_dates",
        "Have the Mets put anyone on the injured list in the last month?",
        {
            "called search_news": lambda r: "search_news" in r.tools_called,
            "answer includes a date": lambda r: any(m in r.answer for m in
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "2026-"]),
        },
    ),
    Case(
        "mixed_question_uses_two_tools",
        "Find a Mets prospect described as having plus power, then give me his stats this season.",
        {
            "searched scouting": lambda r: "search_scouting" in r.tools_called,
            "then pulled stats": lambda r: "season_stats" in r.tools_called,
        },
    ),
]
