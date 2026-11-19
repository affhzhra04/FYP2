"""Tests for src.dash_core on a small synthetic fleet (no database needed)."""
import numpy as np
import pandas as pd
from src import dash_core as C, risk_engine as R


def fleet():
    rows = []
    base = pd.Timestamp("2024-08-09 09:00:00")
    for sid, speed in (("8_9_1", 60.0), ("8_9_2", 100.0)):          # session 2 is speeding
        for i in range(40):
            rows.append((sid, base + pd.Timedelta(seconds=30 * i), 3.0 + i * 1e-3, 101.0 + i * 1e-3,
                         speed, 200.0, 90.0, 1))
    return pd.DataFrame(rows, columns=["vehicle_id", "timestamp", "latitude", "longitude",
                                       "speed", "fuel_level", "heading", "ignition"])


def test_bundle_sessions_and_days():
    b = C.build_bundle(fleet())
    assert len(b.sessions) == 2 and set(b.sessions.vehicle_id) == {"1", "2"}
    days = C.replay_days(b.sessions)
    assert len(days) == 1 and days.iloc[0].vehicles == 2


def test_speeding_session_scores_lower_and_audit_matches():
    b = C.build_bundle(fleet())
    w = dict(R.DEFAULT_WEIGHTS)
    ss = C.session_scores(b, w).set_index("session_id")
    assert ss.loc["8_9_2", "score"] < ss.loc["8_9_1", "score"] == 100.0
    a = C.session_audit(b, "8_9_2", w)
    assert abs(a.score - ss.loc["8_9_2", "score"]) < 1e-9 and len(a.items) == 1


def test_weight_zero_removes_penalty():
    b = C.build_bundle(fleet())
    w = {**R.DEFAULT_WEIGHTS, "speeding": 0.0}
    assert C.session_scores(b, w).set_index("session_id").loc["8_9_2", "score"] == 100.0


def test_live_state_updates_with_time():
    b = C.build_bundle(fleet())
    day = C.day_data(b, C.best_day(C.replay_days(b.sessions)))
    early = C.live_state(b, day, day.t0, dict(R.DEFAULT_WEIGHTS))
    late = C.live_state(b, day, day.t1, dict(R.DEFAULT_WEIGHTS), half_life_h=1000)
    assert early.kpis["events_total"] == 0 and late.kpis["events_total"] == 1
    assert late.leaderboard.iloc[0].vehicle_id == "2"


def test_reports_render():
    b = C.build_bundle(fleet())
    a = C.session_audit(b, "8_9_2", dict(R.DEFAULT_WEIGHTS))
    assert "Driver Safety Report" in C.report_html(a, dict(R.DEFAULT_WEIGHTS), 10.0, b.params)
    pdf = C.report_pdf(a, dict(R.DEFAULT_WEIGHTS), 10.0)
    assert pdf is None or pdf[:4] == b"%PDF"


def test_unreadable_database_gives_friendly_error():
    try:
        C.read_pings("/definitely/not/here.db")
    except C.DataUnavailable as e:
        assert "ingest" in str(e)
    else:
        raise AssertionError("expected DataUnavailable")


def test_session_score_is_normalised_per_100km():
    """Same incidents over twice the distance -> half the risk per 100 km -> higher score."""
    import pandas as pd
    ev = pd.DataFrame([dict(kind="speeding", vehicle_id="1", session_id="a", start=pd.Timestamp("2024-08-09 10:00"),
                            end=pd.Timestamp("2024-08-09 10:05"), duration_s=300.0, magnitude=100.0, extra=0.0)])
    ev["session_id"] = "a"
    expo_short = pd.Series([1.0], index=["a"])
    expo_long = pd.Series([2.0], index=["a"])
    s1 = R.dss(ev, group="session_id", universe=["a"], half_life_h=0, exposure=expo_short).score.iloc[0]
    s2 = R.dss(ev, group="session_id", universe=["a"], half_life_h=0, exposure=expo_long).score.iloc[0]
    assert s2 > s1
    assert abs(R.exposure_100km(10.0) - 0.5) < 1e-9 and abs(R.exposure_100km(300.0) - 3.0) < 1e-9


def test_audit_risk_equals_raw_over_exposure():
    b = C.build_bundle(fleet())
    a = C.session_audit(b, "8_9_2", dict(R.DEFAULT_WEIGHTS))
    assert abs(a.risk - a.meta["raw_risk"] / a.meta["exposure"]) < 1e-9
    assert abs(a.by_kind["points"].sum() - a.risk) < 1e-9
