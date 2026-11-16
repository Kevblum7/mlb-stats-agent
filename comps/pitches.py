"""Pitch comps: "whose slider moves most like Edwin Diaz's?"

Same vector search as the text tools, but each vector is built from
Statcast NUMBERS instead of a text embedding model. That's the lesson:
an embedding is just numbers where distance = similarity. Here YOU decide
what the numbers are (feature engineering), which is where all the
baseball judgment lives.

Build (slow the first time -- pybaseball downloads every pitch in range):
    python -m comps.pitches --start 2026-06-01 --end 2026-06-30   # try a month first
    python -m comps.pitches --start 2026-03-26 --end 2026-09-28   # full season
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np
import pandas as pd

from search.store import DATA, VectorStore, normalize

STORE = DATA / "pitch_comps"
BUILD_HINT = "python -m comps.pitches --start 2026-06-01 --end 2026-06-30"

# Statcast columns averaged per pitcher per pitch type.
# pfx_x / pfx_z = horizontal / vertical movement (feet; we convert to inches).
FEATURES = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z",
            "release_extension", "release_pos_x", "release_pos_z"]
MIN_PITCHES = 50  # fewer than this and the averages are mostly noise


def profiles(pitches: pd.DataFrame, min_pitches: int = MIN_PITCHES) -> pd.DataFrame:
    """One row per (pitcher, pitch type) with average characteristics."""
    df = pitches.dropna(subset=FEATURES + ["pitch_name"]).copy()
    df["pfx_x"] *= 12  # feet -> inches
    df["pfx_z"] *= 12
    g = df.groupby(["pitcher", "player_name", "p_throws", "pitch_name"])
    out = g[FEATURES].mean().round(2)
    out["n"] = g.size()
    return out[out["n"] >= min_pitches].reset_index()


# How much each feature counts. Movement and velocity define how a pitch
# looks to a hitter; release point and extension matter, but less.
WEIGHTS = {"release_speed": 1.0, "release_spin_rate": 0.7, "pfx_x": 1.0, "pfx_z": 1.0,
           "release_extension": 0.5, "release_pos_x": 0.5, "release_pos_z": 0.6}

# Features whose sign depends on which arm throws (glove-side vs arm-side).
MIRROR_FOR_LEFTIES = ["pfx_x", "release_pos_x"]


def feature_matrix(prof: pd.DataFrame) -> np.ndarray:
    """Turn profiles into comparable vectors.

    1. Mirror lefties: a lefty's slider breaks the opposite way, so pfx_x and
       release_pos_x have the opposite sign. Flipping them lets a righty's
       slider match a lefty's mirror-image slider -- same pitch shape,
       other arm. (Use the `throws` filter to restrict to one hand.)
    2. Z-score each feature WITHIN its pitch type, so 1 mph and 100 rpm are
       on the same scale and sliders are compared against sliders.
    3. Weight features by how much they define the pitch (WEIGHTS).
    4. Normalize -> cosine similarity at search time.
    """
    X = prof[FEATURES].astype(float).copy()
    lefty = prof["p_throws"] == "L"
    X.loc[lefty, MIRROR_FOR_LEFTIES] *= -1
    z = X.groupby(prof["pitch_name"]).transform(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1))
    z = z.fillna(0) * pd.Series(WEIGHTS)
    return normalize(z[FEATURES].to_numpy(dtype=np.float32))


def build(pitches: pd.DataFrame) -> VectorStore:
    prof = profiles(pitches)
    items = prof.to_dict(orient="records")
    store = VectorStore(STORE, items, feature_matrix(prof))
    store.save()
    return store


def similar(store: VectorStore, mlbam_id: int, pitch_name: str, k: int = 5,
            throws: str | None = None) -> dict:
    """Pitches most like one pitcher's pitch. throws='R'/'L' limits comps to one hand."""
    target = [i for i, it in enumerate(store.items)
              if it["pitcher"] == mlbam_id and it["pitch_name"].lower() == pitch_name.lower()]
    if not target:
        have = sorted({it["pitch_name"] for it in store.items if it["pitcher"] == mlbam_id})
        return {"error": f"No {pitch_name!r} for pitcher {mlbam_id} in the comps data. "
                         f"Pitch types available for him: {have or 'none (not enough pitches in range)'}"}
    t = store.items[target[0]]
    hits = store.search(
        store.vectors[target[0]], k=k,
        where=lambda it: it["pitch_name"] == t["pitch_name"] and it["pitcher"] != mlbam_id
        and (throws is None or it["p_throws"] == throws.upper()),
    )
    return {"target": t, "similar": [{"similarity": s, **it} for s, it in hits],
            "note": "similarity is cosine on weighted, pitch-type z-scores; lefties mirrored"}


def main() -> None:
    import pybaseball

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args()
    pybaseball.cache.enable()
    with contextlib.redirect_stdout(sys.stderr):
        pitches = pybaseball.statcast(a.start, a.end)
    store = build(pitches)
    print(f"{len(pitches):,} pitches -> {len(store.items):,} pitcher/pitch-type profiles -> {STORE}")


if __name__ == "__main__":
    main()
