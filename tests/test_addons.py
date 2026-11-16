"""Tests for the three search tools. No API key or network: fake embedders
and synthetic Statcast/transaction data."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comps import pitches  # noqa: E402
from news import transactions  # noqa: E402
from scouting import reports  # noqa: E402
from search.store import VectorStore, normalize  # noqa: E402

VOCAB = ["changeup", "command", "power", "slider", "wrist", "injured", "bullpen", "trade"]


def fake_embed(texts):
    return np.array([[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts], dtype=np.float32)


# ------------------------------------------------------------------ store

def test_filter_applies_before_ranking(tmp_path):
    items = [{"team": "Mets"}, {"team": "Yankees"}, {"team": "Mets"}]
    vecs = normalize(np.array([[1, 0], [1, 0.01], [0, 1]], dtype=np.float32))
    store = VectorStore(tmp_path, items, vecs)
    hits = store.search(np.array([1, 0], dtype=np.float32), k=1, where=lambda i: i["team"] == "Yankees")
    assert hits[0][1]["team"] == "Yankees"


# ------------------------------------------------------------------ comps

def statcast_rows(pitcher, name, throws, pitch, speed, spin, px, pz, n=60):
    return [{"pitcher": pitcher, "player_name": name, "p_throws": throws, "pitch_name": pitch,
             "release_speed": speed, "release_spin_rate": spin, "pfx_x": px, "pfx_z": pz,
             "release_extension": 6.5, "release_pos_x": -1.5 if throws == "R" else 1.5,
             "release_pos_z": 5.8} for _ in range(n)]


def make_pitches():
    rows = []
    rows += statcast_rows(1, "Target, Rhp", "R", "Slider", 86, 2500, 0.80, 0.10)
    rows += statcast_rows(2, "Twin, Rhp", "R", "Slider", 86.5, 2480, 0.78, 0.12)    # near-identical
    rows += statcast_rows(3, "Mirror, Lhp", "L", "Slider", 86, 2500, -0.80, 0.10)   # lefty mirror image
    rows += statcast_rows(4, "Different, Rhp", "R", "Slider", 80, 2000, 0.10, -0.40)
    rows += statcast_rows(5, "Fastball, Guy", "R", "4-Seam Fastball", 96, 2300, -0.6, 1.4)
    rows += statcast_rows(6, "Too, Few", "R", "Slider", 86, 2500, 0.8, 0.1, n=10)   # under MIN_PITCHES
    return pd.DataFrame(rows)


def test_profiles_drop_small_samples_and_convert_inches():
    prof = pitches.profiles(make_pitches())
    assert 6 not in set(prof["pitcher"])
    assert prof.loc[prof.pitcher == 1, "pfx_x"].iloc[0] == 9.6  # 0.8 ft -> 9.6 in


def test_comps_rank_twin_and_mirrored_lefty_first(tmp_path, monkeypatch):
    monkeypatch.setattr(pitches, "STORE", tmp_path)
    store = pitches.build(make_pitches())
    out = pitches.similar(store, 1, "slider", k=3)
    names = [s["player_name"] for s in out["similar"]]
    assert set(names[:2]) == {"Twin, Rhp", "Mirror, Lhp"}
    assert names[-1] == "Different, Rhp"
    assert all(s["pitch_name"] == "Slider" for s in out["similar"])  # never compares to fastballs
    righties = pitches.similar(store, 1, "Slider", throws="R")["similar"]
    assert "Mirror, Lhp" not in [s["player_name"] for s in righties]


def test_comps_unknown_pitch_lists_what_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(pitches, "STORE", tmp_path)
    store = pitches.build(make_pitches())
    assert "Slider" in pitches.similar(store, 1, "Changeup")["error"]


# ------------------------------------------------------------------ scouting

def test_long_reports_split_and_keep_player_context():
    rep = {"player": "A. Prospect", "team": "Mets", "position": "RHP", "source": "notes",
           "date": "2026-11-01", "url": "", "text": ("Plus changeup. " * 50) + "\n\n" + ("Command lags. " * 60)}
    chunks = reports.chunk_report(rep)
    assert len(chunks) == 2 and all(c["embed_text"].startswith("A. Prospect") for c in chunks)


def test_scouting_search_filters_by_team(tmp_path, monkeypatch):
    csv_path = tmp_path / "reports.csv"
    csv_path.write_text(
        "player,team,position,source,date,url,text\n"
        "EXAMPLE Player,Mets,RHP,x,2026-01-01,,ignored example row\n"
        "Arm One,Mets,RHP,notes,2026-10-01,,Plus changeup but command is below average\n"
        "Arm Two,Braves,RHP,notes,2026-10-01,,Plus changeup with plus command\n"
        "Bat One,Mets,SS,notes,2026-10-01,,Raw power with swing and miss\n"
    )
    monkeypatch.setattr(reports, "CSV_PATH", csv_path)
    monkeypatch.setattr(reports, "STORE", tmp_path / "idx")
    store = reports.build(embed=fake_embed)
    assert len(store.items) == 3  # EXAMPLE row skipped
    hits = reports.search(store, "changeup command", fake_embed, team="mets")
    assert hits[0]["player"] == "Arm One" and all(h["team"] == "Mets" for h in hits)
    assert "embed_text" not in hits[0]


# ------------------------------------------------------------------ news

RAW = [
    {"id": 1, "date": "2026-11-10", "person": {"id": 9, "fullName": "Pete Alonso"},
     "toTeam": {"name": "New York Mets"}, "typeDesc": "Status Change",
     "description": "New York Mets placed 1B Pete Alonso on the 10-day injured list. Right wrist sprain."},
    {"id": 2, "date": "2026-04-02", "person": {"id": 9, "fullName": "Pete Alonso"},
     "toTeam": {"name": "New York Mets"}, "typeDesc": "Status Change",
     "description": "New York Mets placed 1B Pete Alonso on the 10-day injured list. Right wrist contusion."},
    {"id": 3, "date": "2026-11-12", "person": {"id": 7, "fullName": "Some Reliever"},
     "toTeam": {"name": "Atlanta Braves"}, "fromTeam": {"name": "New York Mets"}, "typeDesc": "Trade",
     "description": "New York Mets traded RHP Some Reliever to Atlanta Braves for bullpen depth."},
    {"id": 3, "date": "2026-11-12", "description": "duplicate id, dropped"},
    {"id": 4, "date": "2026-11-12", "description": ""},
]


def test_news_parsing_dedupes_and_skips_empty():
    items = transactions.to_items(RAW)
    assert [i["id"] for i in items] == [1, 2, 3]
    assert items[2]["from_team"] == "New York Mets"


def test_news_recency_breaks_near_ties(tmp_path, monkeypatch):
    monkeypatch.setattr(transactions, "STORE", tmp_path)
    store = transactions.build(RAW, embed=fake_embed)
    out = transactions.search(store, "wrist injured", fake_embed, player="Alonso", today=date(2026, 11, 16))
    assert [r["id"] for r in out["results"]] == [1, 2]  # November IL move outranks April's
    assert out["data_through"] == "2026-11-12"
    recent = transactions.search(store, "trade", fake_embed, team="Mets", since_days=30, today=date(2026, 11, 16))
    assert 2 not in [r["id"] for r in recent["results"]]  # April filtered out
