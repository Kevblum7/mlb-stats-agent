# MLB Stats Agent

An AI research assistant that answers baseball questions with real data instead
of from memory. Its tools live on an **MCP server**. An agent built with the
**OpenAI Agents SDK** decides which tools to call and in what order, then answers
from the results. A set of **evals** checks both that it got the right numbers
and that it got them the right way.

```text
> Compare Pete Alonso's 2024 and 2025 MLB seasons

Pete Alonso improved significantly from 2024 to 2025. In 2024, he played 162
games with 34 HR, a .240 AVG, .329 OBP, .459 SLG, and .788 OPS. In 2025, also
162 games, he increased to 38 HR, a .272 AVG, .347 OBP, .524 SLG, and .871 OPS.
```

## How it works

```text
question ──> agent (OpenAI Agents SDK, gpt-4.1-mini)
               │  picks a tool, reads the result, repeats (max 10 turns)
               ▼
            MCP client ──stdio──> server.py (MCP server)
                                    ├─ lookup_player   ──> Chadwick player register (pybaseball)
                                    ├─ season_stats    ──> MLB Stats API
                                    ├─ pitch_mix       ──> Statcast (pybaseball)
                                    ├─ similar_pitches ──> local index of Statcast pitch profiles
                                    ├─ search_news     ──> local index of MLB transactions
                                    └─ search_scouting ──> local index of your scouting reports
```

| Tool | What it returns |
|---|---|
| `lookup_player` | Up to 5 players matching a name, with their MLB ids and years played. |
| `season_stats` | One player's regular-season totals. Batting: G, PA, HR, AVG, OBP, SLG, OPS. Pitching: G, IP, ERA, WHIP, K, BB, K%, BB%. |
| `pitch_mix` | Each pitch type a pitcher threw over a date range, with count, usage % and average velocity. |
| `similar_pitches` | The pitches most similar to one pitcher's pitch ("whose slider looks like Edwin Díaz's?"), by velocity, spin, movement and release point. |
| `search_news` | Transactions (IL moves, trades, signings, options) matching a question, ranked by meaning and recency, with dates. |
| `search_scouting` | Scouting-report excerpts matching a description ("plus changeup, shaky command"), filterable by team and position. |

The agent starts `server.py` as a subprocess and talks to it over stdin/stdout.
The model never sees the tool code, only each tool's name, parameter schema and
docstring. The same server also works in other MCP clients, such as Claude
Desktop and Cursor.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add your OPENAI_API_KEY
python -m pytest -q               # no API key needed

python agent.py "Who hit more home runs in 2025, Pete Alonso or Rafael Devers?"
python agent.py                   # interactive chat; remembers earlier questions
python chat_app.py                # Gradio chatbot UI at http://127.0.0.1:7860
python -m evals.run_evals --runs 3
```

You can also call a tool directly, without the model:

```bash
python -c "import server; print(server.season_stats(624413, 2025))"
```

The three search tools read local indexes in `data/`, which isn't committed.
Build each one before using its tool; until then the tool returns the command
to run.

```bash
python -m news.transactions --days 120                         # MLB transactions; refresh to stay current
python -m comps.pitches --start 2026-03-26 --end 2026-09-28    # one season of Statcast, ~4 minutes
python -m scouting.reports    # creates data/scouting/reports.csv; add reports, then run again
```

The news and scouting indexes use OpenAI embeddings (`text-embedding-3-small`).
Scouting reports aren't included because published reports are copyrighted.

## Design decisions

- **Docstrings are prompts.** A docstring is the only description of a tool the
  model gets. Each one says when to call the tool, which id to pass, what it
  returns and what it can't do. For example, `season_stats` states that WAR
  isn't available, so the agent says so instead of guessing.
- **Small outputs.** Tools return a few fields as plain JSON, not DataFrames.
  Everything a tool returns is sent to the model, and a full stats row has
  hundreds of columns.
- **Errors tell the model what to do next.** A missing season, a hitter asked
  for pitching stats, bad dates, or a failed API call each return an
  `{"error": ...}` message that suggests a next step, instead of raising an
  exception.
- **One player id everywhere.** Every tool takes the MLB id (`mlbam_id`) from
  `lookup_player`, so the model never has to choose between id systems.
- **Season stats come from MLB's Stats API, not FanGraphs.** `pybaseball`'s
  FanGraphs scraper now returns empty tables because FanGraphs blocks
  automated requests. The official MLB Stats API is free and returns one player
  per request. The cost is that FanGraphs-only stats (WAR, wRC+, FIP) aren't
  available.
- **Traded players get full-season totals.** The MLB Stats API returns one row
  per team plus a season total, and `season_stats` uses the total.
- **`pitch_mix` refuses ranges over 366 days.** Statcast downloads are slow, so
  the error asks the model to make one call per season and compare.
- **Nothing may print to stdout.** Over stdio, stdout carries the protocol
  messages, and `pybaseball` prints progress text that would break the
  connection. `quiet()` in `server.py` redirects it to stderr.
- **Conversation memory, no compaction yet.** The interactive chat keeps its
  history in an in-memory `SQLiteSession`, so follow-ups like "what about the
  year before?" work. The full history is resent on each question. That's fine
  at chat length because tool outputs are small. If history ever needs trimming,
  what should survive is which players and seasons are being discussed. Stat
  numbers can always be fetched again.

## Search tools

The three search tools share one small vector store (`search/store.py`): each
item is a list of numbers, and items whose numbers point the same way are
similar.

- **`similar_pitches`** builds those numbers from Statcast averages per
  pitcher and pitch type, rather than from a text model. Left-handed pitches
  are mirrored so shapes compare across arms, and each feature is scaled
  within its pitch type. The current index covers the 2026 regular season:
  716,534 pitches, giving 2,584 profiles with at least 50 pitches each.
- **`search_news`** embeds MLB's official transactions feed and ranks results
  by a blend of meaning (75%) and recency (25%). Every response says how
  current the data is. The current index has 19,437 transactions from the last
  120 days.
- **`search_scouting`** embeds scouting reports you collect. Long reports are
  split by paragraph, and each chunk keeps the player's name, team and
  position.

Filters such as team or position are applied before ranking, so a filtered
search still returns its full number of results.

## Evals

`evals/run_evals.py` runs the real agent on 17 questions and grades each answer
with code-based checks. Thirteen cover the stats tools:

- **Single lookups:** a hitter's season, a pitcher's ERA, a traded player's
  full-season total.
- **Multi-step questions:** two players compared, two seasons compared, a
  pitcher's most-used pitch and its velocity.
- **Traps:**
  - a misspelled name ("Alonzo")
  - a name two players share ("Will Smith")
  - a hitter's ERA
  - a stat the tools don't have (WAR)
  - a season that hasn't been played
  - a hitter's pitch mix

Four more (`evals/addon_cases.py`) check that the agent picks the right search
tool: pitch comps, scouting, news, and a question that needs scouting plus
stats. These four only check the process, so they don't depend on what's in
the indexes.

Each case checks the **process** as well as the **outcome**. Outcome checks
compare numbers in the answer against true values. Process checks confirm that
the right tools were called with the right arguments, not just that a tool name
appeared. For example, the two-player case requires one `season_stats` call
with each player's id and the year 2025, and checks that each number in the
answer is attributed to the right player.

### Results

Across three runs of all 17 cases with `gpt-4.1-mini`, **122 of 126 checks
passed**:

- **Stats tools: 102 of 102.**
- **Search tools: 20 of 24.**
  - The scouting-plus-stats question failed its "then pulled stats" check in
    all three runs. There are no scouting reports in the index yet, and the
    agent correctly said so instead of inventing a prospect. The check assumes
    scouting data exists.
  - One news answer listed transactions without their dates.

The evals found two failures during development:

- **An agent bug.** Asked about Alonso's 2027 season, which hasn't been played,
  the agent said he "hit 0 home runs that year." It had treated "no data" as
  zero. Adding one rule to the agent's instructions ("No data is not zero")
  fixed it: the case passed 5 of 5 runs afterwards.
- **An eval bug.** For "Will Smith", the agent correctly asked which player was
  meant, but the check failed. It looked for the words "two players", and the
  agent wrote "two MLB players". Widening the pattern fixed it. String checks
  on free-text answers are brittle in both directions.

Testing the search tools against real data found two more bugs that their unit
tests missed:

- **News search crashed on ties.** With 19,000 real transactions, two results
  could tie on score, and the sort then tried to compare two dicts. It now
  sorts on the score alone.
- **`lookup_player` returned duplicates.** Edwin Díaz appeared twice in his
  five results. Results are now de-duplicated by player id.

### Limitations

- Checks are regexes and number matches. They can't judge whether an answer is
  clear, and they can still pass a wrong answer that happens to contain the
  right number.
- The evals test the whole system at once, including live data sources. An API
  outage looks the same as an agent mistake. Run the tools directly to tell
  which it is.
- The results above use one model. Changing the model or the instructions
  means re-running the evals.
- The search-tool cases check only which tool was called, so they pass even
  when an index is empty. The scouting case passes today with no reports.
- The news team filter matches by substring, so "Mets" also returns
  minor-league affiliates such as the St. Lucie Mets.

## How I built this

Built with Cursor and Claude as pair programmers. I chose the problem and which
tools to build. I wrote first drafts of `season_stats` and `pitch_mix`, and the
AI fixed and finished them. I made the calls where there was a trade-off:

- moving season stats to MLB's Stats API once FanGraphs blocked scraping, and
  accepting the loss of WAR, wRC+ and FIP
- adding conversation memory
- deferring history compaction

I worked through the eval design with the AI: checking the process as well as
the answer, and checking tool arguments rather than only tool names. I wrote the
first numeric check, and the AI wrote the remaining cases. I tested the tools
directly from the terminal.

The search tools are AI-generated. I had them connected and tested against real
data, which found the two bugs above, but I haven't reviewed their design
choices yet: the pitch features and weights, and the recency blend.
