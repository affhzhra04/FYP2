"""Persistent incident / trip reviews and the alert log.

Stored in its own SQLite file (``database/reviews.db``) so re-ingesting telemetry never
wipes a reviewer's decisions. Items are keyed by a stable text id, not a row number:
  incident -> "<session_id>|<kind>|<start timestamp>"
  trip     -> "<session_id>"
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DEFAULT_REVIEW_DB = Path(__file__).resolve().parent.parent / "database" / "reviews.db"
STATUSES = ["Confirmed", "False alarm", "Follow-up needed"]
MYT = timezone(timedelta(hours=8))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    item_id TEXT PRIMARY KEY, item_type TEXT NOT NULL, status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL, channel TEXT NOT NULL,
    mode TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL, sent_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S")


def _connect(path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.executescript(_SCHEMA)
    return con


def incident_key(session_id: str, kind: str, start) -> str:
    return f"{session_id}|{kind}|{pd.Timestamp(start).strftime('%Y-%m-%d %H:%M:%S')}"


def trip_key(session_id: str) -> str:
    return str(session_id)


def save_review(item_id: str, item_type: str, status: str, note: str = "", reviewer: str = "",
                path=DEFAULT_REVIEW_DB) -> None:
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}; use one of {STATUSES}")
    if item_type not in ("incident", "trip"):
        raise ValueError("item_type must be 'incident' or 'trip'")
    with closing(_connect(path)) as con, con:
        con.execute(
            "INSERT INTO reviews(item_id,item_type,status,note,reviewer,updated_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(item_id) DO UPDATE SET status=excluded.status, note=excluded.note, "
            "reviewer=excluded.reviewer, updated_at=excluded.updated_at",
            (item_id, item_type, status, note.strip(), reviewer.strip(), _now()))


def load_reviews(item_type: Optional[str] = None, path=DEFAULT_REVIEW_DB) -> pd.DataFrame:
    with closing(_connect(path)) as con:
        q, args = "SELECT item_id,item_type,status,note,reviewer,updated_at FROM reviews", ()
        if item_type:
            q, args = q + " WHERE item_type=?", (item_type,)
        return pd.read_sql_query(q, con, params=args)


def review_map(item_type: str, path=DEFAULT_REVIEW_DB) -> Dict[str, dict]:
    df = load_reviews(item_type, path)
    return {r.item_id: dict(status=r.status, note=r.note, reviewer=r.reviewer, updated_at=r.updated_at)
            for r in df.itertuples()}


def log_alert(item_id: str, channel: str, mode: str, status: str, message: str, path=DEFAULT_REVIEW_DB) -> None:
    with closing(_connect(path)) as con, con:
        con.execute("INSERT INTO alert_log(item_id,channel,mode,status,message,sent_at) VALUES(?,?,?,?,?,?)",
                    (item_id, channel, mode, status, message, _now()))


def load_alert_log(limit: int = 200, path=DEFAULT_REVIEW_DB) -> pd.DataFrame:
    with closing(_connect(path)) as con:
        return pd.read_sql_query(
            "SELECT sent_at,item_id,channel,mode,status,message FROM alert_log ORDER BY id DESC LIMIT ?",
            con, params=(limit,))
