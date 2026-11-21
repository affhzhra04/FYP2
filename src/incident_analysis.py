"""Incident analysis: turn the detected incidents into answers a fleet manager can act on.

All functions are pure pandas (no Streamlit, no database) so they can be unit-tested.
They expect the incident frame built by the dashboard (``_incident_frame``): one row per incident with
``kind, vehicle_id, session_id, start, duration_s, magnitude, priority, status`` and a ``date`` column.
Timestamps are Malaysia time as ingested.

  by_type          what happens         (count, share of incidents, share of risk points)
  hour_profile     when it happens      (hour of day x type)
  weekday_hour     when it happens      (weekday x hour)
  weekly_trend     is it getting worse  (week x type)
  vehicle_profile  who it happens to    (counts, incidents and risk points per 1,000 km)
  review_outcomes  how reliable is it   (confirmed vs false alarm per type)
  trigger_rules    why it was flagged   (rule in words with the thresholds in force, weight, priority)
  insights         plain-language findings
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import risk_engine as R

KINDS: List[str] = list(R.DEFAULT_WEIGHTS)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LABEL: Dict[str, str] = {
    "speeding": "Speeding",
    "harsh_driving": "Harsh driving",
    "idling": "Excessive idling",
    "route_deviation": "Route deviation",
    "unauthorized_stop": "Unauthorized stop",
    "fuel_theft": "Fuel loss (stationary)",
    "fuel_off_route": "Fuel loss + off-route / unknown stop",
}


def vehicle_sort_key(v: str):
    return (len(str(v)), str(v))


def add_points(ev: pd.DataFrame, weights: Optional[Dict[str, float]] = None,
               speed_limit: float = 90.0) -> pd.DataFrame:
    """Add ``points`` = the incident's raw risk points (same formula as the driver score)."""
    out = ev.copy()
    if out.empty:
        out["points"] = pd.Series(dtype=float)
        return out
    w = {**R.DEFAULT_WEIGHTS, **(weights or {})}
    sev = R.event_severity(out, speed_limit)
    out["points"] = (out["kind"].map(w) / 3.0) * out["kind"].map(R.BASE_PENALTY) * sev.to_numpy()
    return out


def by_type(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per incident type (all types listed, zeros included)."""
    n = ev.groupby("kind").size().reindex(KINDS, fill_value=0)
    high = (ev[ev["priority"] == "High"].groupby("kind").size().reindex(KINDS, fill_value=0)
            if len(ev) else n * 0)
    pts = (ev.groupby("kind")["points"].sum().reindex(KINDS, fill_value=0.0)
           if "points" in ev and len(ev) else pd.Series(0.0, index=KINDS))
    out = pd.DataFrame({"kind": KINDS, "incidents": n.to_numpy(), "high_priority": high.to_numpy(),
                        "points": pts.to_numpy()})
    out["label"] = out["kind"].map(LABEL)
    tot_n, tot_p = out["incidents"].sum(), out["points"].sum()
    out["share_incidents"] = out["incidents"] / tot_n if tot_n else 0.0
    out["share_points"] = out["points"] / tot_p if tot_p else 0.0
    return out


def hour_profile(ev: pd.DataFrame) -> pd.DataFrame:
    """Incidents by hour of day (rows 0..23) and type (columns)."""
    if ev.empty:
        return pd.DataFrame(0, index=range(24), columns=KINDS)
    h = ev["start"].dt.hour
    return (pd.crosstab(h, ev["kind"]).reindex(index=range(24), columns=KINDS, fill_value=0))


def weekday_hour(ev: pd.DataFrame) -> pd.DataFrame:
    """Incident counts, weekday (rows Mon..Sun) by hour of day (columns 0..23)."""
    if ev.empty:
        return pd.DataFrame(0, index=WEEKDAYS, columns=range(24))
    x = pd.crosstab(ev["start"].dt.dayofweek, ev["start"].dt.hour)
    x = x.reindex(index=range(7), columns=range(24), fill_value=0)
    x.index = WEEKDAYS
    return x


def weekly_trend(ev: pd.DataFrame) -> pd.DataFrame:
    """Incidents per week (week start date as index) and type; weeks without incidents are kept as zero."""
    if ev.empty:
        return pd.DataFrame(columns=KINDS)
    wk = ev["start"].dt.to_period("W").dt.start_time
    x = pd.crosstab(wk, ev["kind"]).reindex(columns=KINDS, fill_value=0)
    full = pd.date_range(x.index.min(), x.index.max(), freq="7D")
    return x.reindex(full, fill_value=0)


def vehicle_profile(ev: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Per vehicle: km driven, incidents by type, total, and incidents per 1,000 km.

    ``sessions`` must already be limited to the same vehicles and dates as ``ev`` so the rate compares
    like with like. A vehicle with no km gets a rate of NaN rather than a misleading infinity.
    """
    km = sessions.groupby("vehicle_id")["km"].sum()
    cnt = (pd.crosstab(ev["vehicle_id"], ev["kind"]).reindex(columns=KINDS, fill_value=0)
           if len(ev) else pd.DataFrame(columns=KINDS))
    vehicles = sorted(set(km.index) | set(cnt.index), key=vehicle_sort_key)
    out = pd.DataFrame(index=vehicles)
    out["km"] = km.reindex(vehicles).fillna(0.0)
    for k in KINDS:
        out[k] = cnt[k].reindex(vehicles).fillna(0).astype(int) if k in cnt else 0
    out["incidents"] = out[KINDS].sum(axis=1)
    pts = ev.groupby("vehicle_id")["points"].sum() if ("points" in ev and len(ev)) else pd.Series(dtype=float)
    out["points"] = pts.reindex(vehicles).fillna(0.0)
    has_km = out["km"] > 0
    out["per_1000km"] = np.where(has_km, out["incidents"] / out["km"].where(has_km) * 1000.0, np.nan)
    out["points_per_1000km"] = np.where(has_km, out["points"] / out["km"].where(has_km) * 1000.0, np.nan)
    out.index.name = "vehicle_id"
    return out.reset_index()


def review_outcomes(ev: pd.DataFrame) -> pd.DataFrame:
    """Per type: how many incidents were confirmed, were false alarms, need follow-up or are still open.

    ``confirm_rate`` = confirmed / (confirmed + false alarm), i.e. among incidents with a final decision.
    It is NaN when nobody has decided yet.
    """
    rows = []
    for k in KINDS:
        s = ev.loc[ev["kind"] == k, "status"] if len(ev) else pd.Series(dtype=str)
        c, f = int((s == "Confirmed").sum()), int((s == "False alarm").sum())
        rows.append({"kind": k, "label": LABEL[k], "incidents": int(len(s)), "open": int((s == "Open").sum()),
                     "confirmed": c, "false_alarm": f, "follow_up": int((s == "Follow-up needed").sum()),
                     "confirm_rate": c / (c + f) if (c + f) else np.nan})
    return pd.DataFrame(rows)


def trigger_rules(params: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """The rule behind each incident type in words, with the thresholds and weights currently in force."""
    p = {**R.DEFAULT_PARAMS, **(params or {})}
    w = {**R.DEFAULT_WEIGHTS, **(weights or {})}
    lim = p["speed_limit_kmh"]
    rows = [
        ("speeding", f"Speed above {lim:.0f} km/h", f"High at {lim + 15:.0f} km/h or more, else Medium"),
        ("harsh_driving", f"Speed changes by {p['harsh_mps2']:.2f} m/s² or more between pings that are at most "
                          f"{p['harsh_max_dt_s']:.0f} s apart", "Medium"),
        ("idling", f"Engine on and not moving for {p['idle_sec'] / 60:.0f} min or more", "Low"),
        ("route_deviation", f"Off the vehicle's usual corridor for {p['route_min_minutes']:.0f} min or more "
                            f"(needs {p['route_min_sessions']:.0f}+ trips of history)", "Medium"),
        ("unauthorized_stop", f"Stopped {p['stop_min_minutes']:.0f} min to {p['stop_max_minutes'] / 60:.0f} h at a "
                              f"place that is not home or a known stop; longer stops count as parking",
         "High at 2 h or more, else Medium"),
        ("fuel_theft", f"Fuel drops by {p['fuel_drop_l']:.1f} L or more while parked with the engine off, and "
                       f"stays down", "High"),
        ("fuel_off_route", f"A fuel loss within {p['corr_window_min']:.0f} min of a route deviation or "
                           f"unauthorized stop on the same trip", "High"),
    ]
    return pd.DataFrame([{"kind": k, "Incident": LABEL[k], "Flagged when": rule, "Priority": pr,
                          "Weight": w[k]} for k, rule, pr in rows])


def insights(ev: pd.DataFrame, sessions: pd.DataFrame) -> List[str]:
    """Plain-language findings for the page summary. Returns [] when there is nothing to report."""
    if ev.empty:
        return []
    out: List[str] = []
    t = by_type(ev)
    top = t.sort_values("incidents", ascending=False).iloc[0]
    out.append(f"**{top['label']}** is the most frequent incident type: {int(top['incidents']):,} of "
               f"{int(t['incidents'].sum()):,} incidents ({top['share_incidents']:.0%}).")

    if "points" in ev and t["points"].sum() > 0:
        costly = t.sort_values("share_points", ascending=False).iloc[0]
        if costly["kind"] != top["kind"]:
            out.append(f"**{costly['label']}** contributes the largest share of risk points "
                       f"({costly['share_points']:.0%}) from {costly['share_incidents']:.0%} of incidents.")
        else:
            out.append(f"It also contributes the largest share of risk points ({costly['share_points']:.0%}).")

    hp = hour_profile(ev).sum(axis=1)
    if hp.max() > 0:
        h = int(hp.idxmax())
        out.append(f"Incident volume peaks between **{h:02d}:00 and {(h + 1) % 24:02d}:00** "
                   f"({int(hp.max()):,} incidents).")

    vp = vehicle_profile(ev, sessions)
    vp = vp[vp["per_1000km"].notna() & (vp["km"] >= 100.0)]     # too little driving to compare fairly
    if len(vp) >= 2:
        r = vp.sort_values("per_1000km", ascending=False).iloc[0]
        mix = r[KINDS].astype(float)
        lead = mix.idxmax()
        note = f", of which {mix[lead] / mix.sum():.0%} {LABEL[lead].lower()}" if mix.sum() and mix[lead] / mix.sum() >= 0.6 else ""
        out.append(f"**Vehicle {r['vehicle_id']}** has the highest incident rate at {r['per_1000km']:.1f} per 1,000 km "
                   f"({int(r['incidents']):,} incidents over {r['km']:,.0f} km{note}).")

    ro = review_outcomes(ev)
    c, f = int(ro["confirmed"].sum()), int(ro["false_alarm"].sum())
    if c + f:
        out.append(f"Of {c + f} incidents with a final decision, **{f}** were false alarms ({f / (c + f):.0%}).")
    return out
