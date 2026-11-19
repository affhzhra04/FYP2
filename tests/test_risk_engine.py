"""Unit tests for src.risk_engine (synthetic, deterministic)."""
import numpy as np
import pandas as pd
import pytest
from src import risk_engine as R


def make(rows, vid="8_9_8"):
    df = pd.DataFrame(rows, columns=["timestamp", "latitude", "longitude", "speed", "fuel_level", "ignition"])
    df.insert(0, "vehicle_id", vid)
    return df


def t(s):  # seconds offset -> timestamp
    return pd.Timestamp("2024-08-09 10:00:00") + pd.Timedelta(seconds=s)


def test_session_vehicle_split():
    df = R.prepare(make([(t(0), 3.0, 101.0, 0, 50, 0), (t(10), 3.0, 101.0, 0, 50, 0)], vid="5_6_8"))
    assert df.session_id.iloc[0] == "5_6_8" and df.vehicle_id.iloc[0] == "8"


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        R.prepare(pd.DataFrame({"vehicle_id": ["1_1_1"], "timestamp": ["2024-01-01"]}))


def test_empty_raises():
    with pytest.raises(ValueError):
        R.prepare(pd.DataFrame(columns=["vehicle_id", "timestamp", "latitude", "longitude", "speed", "fuel_level", "ignition"]))


def test_speeding_rows_collapse_to_one_event():
    rows = [(t(i * 10), 3.0 + i * 1e-3, 101.0, v, 50, 1) for i, v in enumerate([80, 95, 100, 98, 80])]
    _, ev = R.detect_all(make(rows))
    sp = ev[ev.kind == "speeding"]
    assert len(sp) == 1 and sp.magnitude.iloc[0] == 100


def test_harsh_requires_dense_pings():
    dense = make([(t(0), 3.0, 101.0, 80, 50, 1), (t(3), 3.0, 101.0, 30, 50, 1)])   # -13.9 m/s^2... harsh
    sparse = make([(t(0), 3.0, 101.0, 80, 50, 1), (t(60), 3.0, 101.0, 30, 50, 1)])  # 60 s gap -> ignored
    assert (R.detect_all(dense)[1].kind == "harsh_driving").sum() == 1
    assert (R.detect_all(sparse)[1].kind == "harsh_driving").sum() == 0


def test_idling_needs_five_minutes():
    short = make([(t(i * 60), 3.0, 101.0, 0, 50, 1) for i in range(4)])     # 3 min
    long_ = make([(t(i * 60), 3.0, 101.0, 0, 50, 1) for i in range(8)])     # 7 min
    assert (R.detect_all(short)[1].kind == "idling").sum() == 0
    assert (R.detect_all(long_)[1].kind == "idling").sum() == 1


def test_fuel_theft_stationary_persistent_only():
    theft = make([(t(i * 30), 3.0, 101.0, 0, f, 0) for i, f in enumerate([100, 100, 90, 90, 90])])
    slosh = make([(t(i * 30), 3.0, 101.0, 0, f, 0) for i, f in enumerate([100, 100, 90, 100, 100])])
    moving = make([(t(i * 30), 3.0, 101.0, 60, f, 1) for i, f in enumerate([100, 100, 90, 90, 90])])
    assert (R.detect_all(theft)[1].kind == "fuel_theft").sum() == 1
    assert (R.detect_all(slosh)[1].kind == "fuel_theft").sum() == 0
    assert (R.detect_all(moving)[1].kind == "fuel_theft").sum() == 0


def test_fuel_theft_ignores_long_gap():
    gap = make([(t(0), 3.0, 101.0, 0, 100, 0), (t(7200), 3.0, 101.0, 0, 80, 0), (t(7230), 3.0, 101.0, 0, 80, 0)])
    assert (R.detect_all(gap)[1].kind == "fuel_theft").sum() == 0


def test_dss_bounds_monotonic_and_recovery():
    ev = pd.DataFrame({"event_id": [1], "kind": ["fuel_theft"], "vehicle_id": ["8"], "session_id": ["8_9_8"],
                       "start": [t(0)], "end": [t(0)], "duration_s": [0.0], "magnitude": [30.0],
                       "lat": [3.0], "lon": [101.0]})
    now = R.dss(ev, as_of=t(60), universe=["8"]).score.iloc[0]
    later = R.dss(ev, as_of=t(60 + 12 * 3600), universe=["8"]).score.iloc[0]
    assert 0 < now < 100 and later > now                 # score recovers as event ages
    hi = R.dss(ev, as_of=t(60), weights={"fuel_theft": 5}, universe=["8"]).score.iloc[0]
    lo = R.dss(ev, as_of=t(60), weights={"fuel_theft": 1}, universe=["8"]).score.iloc[0]
    assert lo > hi                                       # higher weight -> lower score


def test_clean_vehicle_scores_100():
    s = R.dss(pd.DataFrame(columns=R.EVENT_COLUMNS), universe=["1"])
    assert s.score.iloc[0] == 100.0
