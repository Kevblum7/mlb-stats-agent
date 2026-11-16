"""A tiny vector store shared by all three search tools.

The idea that ties them together: an EMBEDDING is any list of numbers where
distance means similarity. Scouting reports and transactions get their
vectors from a text embedding model; pitches get theirs straight from
Statcast numbers (velo, spin, movement). The search math is identical.

Each store is a folder:
    items.json     one dict of metadata per row (player, team, date, text...)
    vectors.npy    one vector per row, same order
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def normalize(m: np.ndarray) -> np.ndarray:
    """Unit length, so a dot product = cosine similarity."""
    return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(min=1e-12)


class VectorStore:
    def __init__(self, path: Path, items: list[dict], vectors: np.ndarray):
        self.path, self.items, self.vectors = Path(path), items, vectors

    # ---------------------------------------------------------------- disk
    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self.vectors)
        (self.path / "items.json").write_text(json.dumps(self.items, default=str))

    @classmethod
    def load(cls, path: Path, build_hint: str) -> "VectorStore":
        if not (Path(path) / "items.json").exists():
            raise FileNotFoundError(f"No index at {path}. Build it first: {build_hint}")
        return cls(path, json.loads((Path(path) / "items.json").read_text()),
                   np.load(Path(path) / "vectors.npy"))

    # -------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int = 5,
               where: Callable[[dict], bool] | None = None) -> list[tuple[float, dict]]:
        """Top-k (score, item) by cosine similarity.

        `where` filters on metadata BEFORE ranking ("only Mets prospects",
        "only sliders"). Filtering first matters: if you rank everything and
        filter after, the top 5 might all get filtered out."""
        mask = np.array([where(it) for it in self.items]) if where else np.ones(len(self.items), bool)
        if not mask.any():
            return []
        idx = np.flatnonzero(mask)
        scores = self.vectors[idx] @ normalize(query)
        best = np.argsort(-scores)[:k]
        return [(round(float(scores[b]), 3), self.items[idx[b]]) for b in best]

    def scored(self, query: np.ndarray, where: Callable[[dict], bool] | None = None) -> list[tuple[float, dict]]:
        """Every matching item with its similarity, unsorted -- for callers
        that combine similarity with another signal (e.g. recency)."""
        q = normalize(query)
        return [(float(v @ q), it) for v, it in zip(self.vectors, self.items) if not where or where(it)]


def text_embedder() -> Callable[[list[str]], np.ndarray]:
    """OpenAI text embeddings. Use the SAME model to build and to query."""
    import os
    from openai import OpenAI

    client, model = OpenAI(), os.environ.get("EMBED_MODEL", "text-embedding-3-small")

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 100):
            out.extend(d.embedding for d in client.embeddings.create(model=model, input=texts[i:i + 100]).data)
        return np.array(out, dtype=np.float32)

    return embed
