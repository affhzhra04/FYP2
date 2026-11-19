"""Tests for the review store, alert rules and driver performance aggregation."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import alerts as A  # noqa: E402
from src import dash_core as C  # noqa: E402
from src import performance as P  # noqa: E402
from src import review_store as RS  # noqa: E402
from src import risk_engine as R  # noqa: E402


def _db():
    return Path(tempfile.mkdtemp()) / "reviews.db"


def test_review_save_update_and_reload():
    db = _db()
    key = RS.incident_key("8_7_9", "speeding", pd.Timestamp("2024-08-07 15:05:57"))
    assert key == "8_7_9|speeding|2024-08-07 15:05:57"
    RS.save_review(key, "incident", "Confirmed", "driver warned", "Aina", path=db)
    RS.save_review(key, "incident", "False alarm", "  road works ", "Aina", path=db)   # update, not duplicate
    m = RS.review_map("incident", path=db)
    assert len(m) == 1 and m[key]["status"] == "False alarm" and m[key]["note"] == "road works"
    assert RS.review_map("trip", path=db) == {}


def test_review_rejects_bad_input():
    db = _db()
    for args in (("x", "incident", "Maybe"), ("x", "other", "Confirmed")):
        try:
            RS.save_review(*args, path=db)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_alert_log_roundtrip():
    db = _db()
    RS.log_alert("k1", "Telegram", "SIMULATED", "DELIVERED", "msg", path=db)
    log = RS.load_alert_log(path=db)
    assert len(log) == 1 and log.iloc[0]["mode"] == "SIMULATED"


def test_priority_rules():
    ev = pd.DataFrame({
        "kind": ["idling", "speeding", "speeding", "fuel_off_route", "unauthorized_stop", "unauthorized_stop", "route_deviation"],
        "magnitude": [0, 95, 110, 5, 0, 0, 3],
        "duration_s": [600, 60, 60, 60, 3600, 9000, 600]})
    assert A.priority(ev, 90).tolist() == ["Low", "Medium", "High", "High", "Medium", "High", "Medium"]


def test_alert_text_is_markdown_safe():
    t = A.alert_text("8", "Fuel loss", "1.3 L lost on trip 8_7_9 *now*", "2024-08-07 15:05:57")
    assert "_" not in t and "*" not in t and "Vehicle: 8" in t


def test_dispatch_never_raises():
    mode, status, _ = A.dispatch("test")
    assert isinstance(mode, str) and isinstance(status, str)


def _scored(rows):
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"])
    df["date"] = df["start"].dt.date
    df["pings"] = 100
    df["events"] = 1
    df["tier"] = df["score"].map(R.tier)
    df["session_id"] = [f"s{i}" for i in range(len(df))]
    return df


def test_driver_table_ranks_and_flags_limited_data():
    scored = _scored([
        dict(vehicle_id="1", start="2024-05-01", km=200.0, score=90.0),
        dict(vehicle_id="1", start="2024-05-02", km=200.0, score=80.0),
        dict(vehicle_id="2", start="2024-05-01", km=100.0, score=20.0),
        dict(vehicle_id="3", start="2024-05-01", km=10.0, score=5.0),       # did not drive: excluded
    ])
    ev = pd.DataFrame(columns=R.EVENT_COLUMNS)
    t = P.driver_table(scored, ev, dict(R.DEFAULT_WEIGHTS), 10.0)
    assert t["vehicle_id"].tolist() == ["2", "1"]                      # worst first
    assert (t["Data"] == "Limited data").all() and abs(t.set_index("vehicle_id").loc["1", "Mean DSS"] - 85) < 1e-9


def test_trend_compares_last_two_months_with_data():
    rows = [dict(vehicle_id="1", start="2024-01-10", km=100.0, score=40.0),
            dict(vehicle_id="1", start="2024-01-12", km=100.0, score=50.0),     # Jan mean 45
            dict(vehicle_id="1", start="2024-03-05", km=100.0, score=60.0),     # only one day in March: ignored
            dict(vehicle_id="1", start="2024-05-01", km=100.0, score=70.0),
            dict(vehicle_id="1", start="2024-05-02", km=100.0, score=80.0)]     # May mean 75
    t = P.driver_table(_scored(rows), pd.DataFrame(columns=R.EVENT_COLUMNS), dict(R.DEFAULT_WEIGHTS), 10.0)
    assert abs(t.iloc[0]["Trend (pts)"] - 30.0) < 1e-9
