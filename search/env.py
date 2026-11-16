"""Load .env into os.environ (no extra dependency).

The MCP server needs this too: when the agent launches server.py as a
subprocess, the MCP client passes only a minimal, safe set of environment
variables -- NOT your OPENAI_API_KEY. So the server loads .env itself."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
