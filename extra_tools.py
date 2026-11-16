"""Adds three search tools to your MCP server with one line.

In server.py, just above `if __name__ == "__main__":`, add:

    from extra_tools import register
    register(mcp)

Tools added:
    similar_pitches   pitch comps from Statcast numbers (vector search on numbers)
    search_scouting   RAG over your scouting reports (text, filter by team/position)
    search_news       RAG over MLB transactions, blending meaning + recency

Each tool loads its index on first use and keeps it in memory. If an index
hasn't been built yet, the tool returns an error telling the model (and you)
the exact command to build it -- raw exceptions would reach the client only
as a generic "Error executing tool".
"""

from __future__ import annotations

from functools import lru_cache

from search.env import load_env
from search.store import VectorStore


@lru_cache(maxsize=1)
def _embed():
    from search.store import text_embedder
    return text_embedder()


@lru_cache(maxsize=None)
def _store(name: str) -> VectorStore:
    from comps import pitches
    from news import transactions
    from scouting import reports

    mod = {"comps": pitches, "scouting": reports, "news": transactions}[name]
    return VectorStore.load(mod.STORE, mod.BUILD_HINT)


def register(mcp) -> None:
    load_env()  # the MCP client doesn't pass OPENAI_API_KEY to this subprocess

    @mcp.tool()
    def similar_pitches(pitcher_mlbam_id: int, pitch_name: str, k: int = 5,
                        throws: str | None = None) -> dict:
        """Find pitches that look most like one pitcher's pitch (pitch comps).

        Compares velocity, spin, horizontal/vertical movement, extension and
        release point from Statcast. Get pitcher_mlbam_id from lookup_player.
        pitch_name uses Statcast names, e.g. '4-Seam Fastball', 'Sinker',
        'Cutter', 'Slider', 'Sweeper', 'Curveball', 'Knuckle Curve',
        'Changeup', 'Split-Finger'. If the name is wrong, the error lists the
        pitcher's actual pitch types. Lefties are mirrored so shapes compare
        across arms; pass throws='R' or 'L' to limit comps to one hand.
        Report similarity scores and the key numbers (velo, movement in inches).
        """
        from comps.pitches import similar
        try:
            return similar(_store("comps"), pitcher_mlbam_id, pitch_name, k=k, throws=throws)
        except FileNotFoundError as e:
            return {"error": str(e)}

    @mcp.tool()
    def search_scouting(query: str, team: str | None = None, position: str | None = None,
                        k: int = 5) -> dict:
        """Search scouting reports by meaning, e.g. 'plus changeup but below-average
        command' or 'raw power with swing-and-miss concerns'.

        Optional filters: team (e.g. 'Mets'), position (e.g. 'RHP', 'SS').
        Returns matching report excerpts with player, team, source and date.
        Quote or paraphrase only what the excerpts say and cite the source;
        use the stats tools for numbers. If nothing relevant comes back, say so.
        """
        from scouting.reports import search
        try:
            return {"results": search(_store("scouting"), query, _embed(), team=team, position=position, k=k)}
        except FileNotFoundError as e:
            return {"error": str(e)}

    @mcp.tool()
    def search_news(query: str, team: str | None = None, player: str | None = None,
                    since_days: int | None = None, k: int = 8) -> dict:
        """Search MLB transactions: injured-list moves, activations, trades,
        signings, options, DFAs. Ranks by meaning plus recency.

        Examples: query='injured list wrist', player='Alonso';
        query='bullpen moves', team='Mets', since_days=30.
        Always give the date of each transaction and the 'data_through' date,
        so the user knows how current the information is.
        """
        from news.transactions import search
        try:
            return search(_store("news"), query, _embed(), team=team, player=player,
                          since_days=since_days, k=k)
        except FileNotFoundError as e:
            return {"error": str(e)}
