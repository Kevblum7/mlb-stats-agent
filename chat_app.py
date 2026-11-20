"""Gradio chat UI for the MLB Stats Agent.

    source .venv/bin/activate
    pip install -r requirements.txt
    python chat_app.py

Opens http://127.0.0.1:7860 — same agent + MCP tools as agent.py.
"""

from __future__ import annotations

import atexit
import asyncio

import gradio as gr
from agents import SQLiteSession

import agent as app

_server_cm = None
_server = None
_session = SQLiteSession("gradio-chat")
_lock = asyncio.Lock()


async def ensure_server():
    """Start server.py once and keep it for the life of this process."""
    global _server_cm, _server
    async with _lock:
        if _server is None:
            app.load_env()
            _server_cm = app.stats_server()
            _server = await _server_cm.__aenter__()
        return _server


async def shutdown_server() -> None:
    global _server_cm, _server
    if _server_cm is not None:
        await _server_cm.__aexit__(None, None, None)
        _server_cm = _server = None


def _atexit_cleanup() -> None:
    if _server_cm is None:
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(shutdown_server())
        loop.close()
    except Exception:
        pass


atexit.register(_atexit_cleanup)


async def respond(message: str, history: list) -> str:
    question = (message or "").strip()
    if not question:
        return "Ask a baseball question."
    server = await ensure_server()
    answer = await app.ask(question, server, _session)
    return answer or "(no answer)"


EXAMPLES = [
    "Compare Pete Alonso's 2024 and 2025 seasons",
    "What was Paul Skenes' ERA in 2025?",
    "Whose slider looks most like Edwin Diaz's?",
    "Which Mets pitching prospects have a good changeup but shaky command?",
    "Have the Mets made any roster moves in the last month?",
]


def build_ui() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=respond,
        title="MLB Stats Agent",
        description=(
            "Front-office research assistant. Uses live stats, Statcast pitch comps, "
            "transaction news, and your local scouting reports — not model memory."
        ),
        examples=EXAMPLES,
        chatbot=gr.Chatbot(height=480, placeholder="Ask about players, seasons, pitch mixes, comps, news…"),
        textbox=gr.Textbox(placeholder="e.g. Who hit more home runs in 2025, Alonso or Devers?"),
    )


if __name__ == "__main__":
    app.load_env()
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Sans"),
    )
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=theme,
        css=".gradio-container { max-width: 880px !important; }",
    )
