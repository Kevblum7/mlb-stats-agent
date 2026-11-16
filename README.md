# MLB Stats Agent

An AI assistant that answers baseball questions with real data. An **MCP
server** exposes MLB stats tools (via `pybaseball`), and an **OpenAI Agents SDK**
agent decides which tools to call, in what order, to answer an open-ended
question. Evals check that it calls the right tools and gets the numbers right.

```
you ──question──> agent (OpenAI Agents SDK)
                    │  decides which tools to call, reads results, repeats
                    ▼
                 MCP client ──stdio──> server.py (MCP server)
                                        ├─ lookup_player
                                        ├─ season_stats
                                        └─ pitch_mix
                                             └─> pybaseball (FanGraphs, Statcast)
```

## Setup

```bash
cd mlb-stats-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your OpenAI key
python -m pytest -q             # MCP plumbing test, no API key needed
```

## Today's build order

1. **Read `server.py`'s worked example** (`lookup_player`) and the `quiet()` note.
2. **Write `season_stats`** (about 45 minutes). Test it without the agent:
   `python -c "import server; print(server.season_stats(<id>, 2025))"`
3. **Write `pitch_mix`** (about 45 minutes), same way.
4. **Run the agent:** `python agent.py "Compare Pete Alonso's 2024 and 2025 seasons"`.
   Watch which tools it calls. If it calls the wrong one, improve the
   **docstring**: that's the tool's only instruction to the model.
5. **Write ~10 evals** in `evals/run_evals.py` and run them 3 times
   (`--runs 3`). Fix what fails, then write down what you changed and why.
6. **Fill in "Results" below.** That section is your interview story.

## Lessons this project teaches

- **MCP = a standard plug between AI apps and tools.** The same server works in
  the Agents SDK, Claude Desktop and Cursor. The client only sees each tool's name,
  parameter schema and docstring.
- **Tool design is prompt design.** Small outputs, clear errors that say what
  to try next, docstrings written for an LLM reader.
- **stdio gotcha:** stdout is the protocol channel, so anything a library
  prints there breaks the connection (see `quiet()` in `server.py`).
- **Evals check process and outcome:** the right tools in the right order, and
  numbers that match a trusted source.

## Results

_Fill this in: eval pass rate across 3 runs, one failure you found, and what fixed it._
