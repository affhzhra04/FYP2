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


def test_parking_is_not_an_unauthorized_stop():
    """Stops of 6 h or more, and vehicle-days with no movement, are parking, not unplanned stops."""
    import numpy as np
    import pandas as pd
    from src import risk_engine as R

    def day(sid, minutes_stop, drive_first=True):
        t0 = pd.Timestamp("2024-03-01 09:00:00")
        rows = []
        n_drive = 60 if drive_first else 0
        for i in range(n_drive):                       # about 20 km of driving
            rows.append((t0 + pd.Timedelta(seconds=30 * i), 3.10 + 0.002 * i, 101.60, 40.0, 1, 100.0))
        t1 = t0 + pd.Timedelta(seconds=30 * n_drive)
        for m in range(0, minutes_stop + 1, 5):       # then stationary at a place no other session uses
            rows.append((t1 + pd.Timedelta(minutes=m), 3.30, 101.75, 0.0, 0, 100.0))
        d = pd.DataFrame(rows, columns=["timestamp", "latitude", "longitude", "speed", "ignition", "fuel_level"])
        d["vehicle_id"] = 1
        d["session_id"] = sid
        return d

    p = R.DEFAULT_PARAMS
    for minutes, drive, expect in [(120, True, 1), (480, True, 0), (120, False, 0)]:
        pings = day("s", minutes, drive)
        prep = R.prepare(pings)
        got = R.detect_unauthorized_stops(prep, p, homes=pd.DataFrame(
            {"vehicle_id": ["1"], "home_lat": [0.0], "home_lon": [0.0]}))
        assert len(got) == expect, (minutes, drive, len(got))
