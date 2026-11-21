"""Tests for src.incident_analysis (pure aggregation, no database or Streamlit)."""
import numpy as np
import pandas as pd

from src import incident_analysis as IA
from src import risk_engine as R


def frame():
    """Five incidents on two vehicles. Vehicle 1 drove 500 km, vehicle 2 drove 100 km."""
    rows = [
        # kind, vehicle, session, start, duration_s, magnitude, priority, status
        ("speeding", "1", "8_1_1", "2024-08-05 14:10:00", 60, 110.0, "High", "Confirmed"),
        ("speeding", "1", "8_1_1", "2024-08-05 14:40:00", 30, 95.0, "Medium", "False alarm"),
        ("idling", "1", "8_1_1", "2024-08-06 09:00:00", 900, 0.0, "Low", "Open"),
        ("fuel_theft", "2", "8_2_1", "2024-08-12 23:30:00", 120, 12.0, "High", "Confirmed"),
        ("fuel_off_route", "2", "8_2_1", "2024-08-12 23:30:00", 120, 12.0, "High", "Open"),
    ]
    ev = pd.DataFrame(rows, columns=["kind", "vehicle_id", "session_id", "start", "duration_s", "magnitude",
                                     "priority", "status"])
    ev["start"] = pd.to_datetime(ev["start"])
    ev["date"] = ev["start"].dt.date
    return ev


def sessions():
    return pd.DataFrame({"vehicle_id": ["1", "1", "2"], "km": [300.0, 200.0, 100.0]})


def test_by_type_lists_every_type_and_shares_sum_to_one():
    t = IA.by_type(IA.add_points(frame()))
    assert list(t["kind"]) == IA.KINDS
    assert t.set_index("kind").loc["speeding", "incidents"] == 2
    assert t.set_index("kind").loc["harsh_driving", "incidents"] == 0
    assert abs(t["share_incidents"].sum() - 1) < 1e-9 and abs(t["share_points"].sum() - 1) < 1e-9
    assert t.set_index("kind").loc["fuel_theft", "high_priority"] == 1


def test_points_match_the_scoring_formula():
    ev = IA.add_points(frame(), speed_limit=90.0)
    sev = R.event_severity(frame(), 90.0)
    expected = (R.DEFAULT_WEIGHTS["speeding"] / 3.0) * R.BASE_PENALTY["speeding"] * sev.iloc[0]
    assert abs(ev.loc[0, "points"] - expected) < 1e-9
    # raising a weight raises that type's points only
    heavier = IA.add_points(frame(), {"speeding": 6.0})
    assert abs(heavier.loc[0, "points"] - 2 * ev.loc[0, "points"]) < 1e-9
    assert abs(heavier.loc[2, "points"] - ev.loc[2, "points"]) < 1e-9


def test_hour_profile_has_24_rows_and_counts_by_start_hour():
    h = IA.hour_profile(frame())
    assert list(h.index) == list(range(24))
    assert h.loc[14, "speeding"] == 2 and h.loc[23, "fuel_theft"] == 1 and h.values.sum() == 5


def test_weekday_hour_places_incidents_on_the_right_day():
    m = IA.weekday_hour(frame())
    assert m.shape == (7, 24)
    assert m.loc["Mon", 14] == 2                     # 2024-08-05 was a Monday
    assert m.loc["Tue", 9] == 1                      # 2024-08-06, idling
    assert m.loc["Mon", 23] == 2                     # 2024-08-12 was also a Monday
    assert m.values.sum() == 5


def test_weekly_trend_keeps_empty_weeks():
    w = IA.weekly_trend(frame())
    assert list(w.index) == list(pd.date_range("2024-08-05", "2024-08-12", freq="7D"))
    assert w.sum().sum() == 5
    ev = frame()
    ev.loc[3:4, "start"] = pd.Timestamp("2024-08-26 10:00:00")     # leave a gap week
    assert len(IA.weekly_trend(ev)) == 4 and IA.weekly_trend(ev).iloc[1].sum() == 0


def test_vehicle_rate_is_per_1000_km_and_handles_zero_km():
    vp = IA.vehicle_profile(frame(), sessions()).set_index("vehicle_id")
    assert vp.loc["1", "incidents"] == 3 and abs(vp.loc["1", "per_1000km"] - 6.0) < 1e-9     # 3 / 500 km
    assert vp.loc["2", "incidents"] == 2 and abs(vp.loc["2", "per_1000km"] - 20.0) < 1e-9    # 2 / 100 km
    s = pd.DataFrame({"vehicle_id": ["1", "3"], "km": [500.0, 0.0]})
    vp0 = IA.vehicle_profile(frame(), s).set_index("vehicle_id")
    assert np.isnan(vp0.loc["3", "per_1000km"]) and np.isnan(vp0.loc["2", "per_1000km"])


def test_review_outcomes_confirm_rate_ignores_undecided():
    ro = IA.review_outcomes(frame()).set_index("kind")
    assert ro.loc["speeding", "confirmed"] == 1 and ro.loc["speeding", "false_alarm"] == 1
    assert ro.loc["speeding", "confirm_rate"] == 0.5
    assert np.isnan(ro.loc["idling", "confirm_rate"])          # only an Open incident, nothing decided
    assert ro.loc["fuel_theft", "confirm_rate"] == 1.0


def test_trigger_rules_follow_the_thresholds_in_force():
    r = IA.trigger_rules({"speed_limit_kmh": 80.0, "idle_sec": 600.0}, {"speeding": 4.5}).set_index("kind")
    assert "80 km/h" in r.loc["speeding", "Flagged when"] and "95 km/h" in r.loc["speeding", "Priority"]
    assert "10 min" in r.loc["idling", "Flagged when"]
    assert r.loc["speeding", "Weight"] == 4.5 and len(r) == len(IA.KINDS)


def test_insights_name_the_top_type_peak_hour_and_riskiest_vehicle():
    txt = " ".join(IA.insights(IA.add_points(frame()), sessions()))
    assert "Speeding" in txt                       # most common (2 of 5)
    assert "14:00 and 15:00" in txt
    assert "Vehicle 2" in txt                      # 20 per 1,000 km beats 6
    assert "false alarms" in txt


def test_everything_copes_with_no_incidents():
    ev = frame().iloc[0:0]
    assert IA.insights(ev, sessions()) == []
    assert IA.by_type(ev)["incidents"].sum() == 0
    assert IA.hour_profile(ev).values.sum() == 0 and IA.weekday_hour(ev).values.sum() == 0
    assert IA.weekly_trend(ev).empty
    assert IA.vehicle_profile(ev, sessions())["incidents"].sum() == 0
    assert IA.review_outcomes(ev)["incidents"].sum() == 0
    assert "points" in IA.add_points(ev).columns


def test_vehicle_points_rate_and_dominant_type_note():
    ev = IA.add_points(frame())
    vp = IA.vehicle_profile(ev, sessions()).set_index("vehicle_id")
    expected = ev[ev["vehicle_id"] == "2"]["points"].sum() / 100.0 * 1000.0
    assert abs(vp.loc["2", "points_per_1000km"] - expected) < 1e-9
    no_pts = IA.vehicle_profile(frame(), sessions()).set_index("vehicle_id")      # no points column supplied
    assert no_pts.loc["1", "points"] == 0.0
    idle_heavy = pd.concat([frame()] + [frame().iloc[[2]]] * 20, ignore_index=True)
    idle_heavy["vehicle_id"] = np.where(idle_heavy.index >= 5, "1", idle_heavy["vehicle_id"])
    txt = " ".join(IA.insights(IA.add_points(idle_heavy), sessions()))
    assert "excessive idling" in txt.split("Vehicle")[1]                    # the highest-rate finding names the mix
