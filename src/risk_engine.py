"""Episode-based Driver Risk Engine (FYP2).

Fixes and extends the FYP1 pipeline:
  * File name ``month_day_vehicle`` -> real vehicle id + session (vehicle-day).
  * Row-level flags are collapsed into EVENTS (episodes) to avoid alert flooding.
  * Per-vehicle home base (no hard-coded KL coordinate).
  * New detectors: route deviation (learned corridor), unauthorized stop.
  * Harsh-driving guarded by a maximum ping interval (sparse pings are unreliable).
  * Fuel theft with gap guard + persistence, correlated with route/stop events.
  * One tunable, time-decayed DSS:  DSS(t) = 100 * exp(-R(t) / R0)
        R(t) = sum_e  w_k * s_e * 0.5 ** ((t - t_e) / half_life)

All logic is deterministic and explainable. Timestamps are treated as given
(naive); the UI layer is responsible for any timezone presentation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

EARTH_KM = 6371.0088

DEFAULT_PARAMS: Dict[str, float] = {
    "speed_limit_kmh": 90.0,
    "harsh_mps2": 2.78,          # ~10 km/h/s, as in the FYP1 methodology
    "harsh_max_dt_s": 15.0,      # acceleration is only meaningful on dense pings
    "idle_sec": 300.0,
    "stop_min_minutes": 30.0,
    "stop_cell_deg": 0.005,      # ~550 m grid for "known places"
    "stop_known_sessions": 5,    # a place visited by >=5 sessions is a known stop
    "stop_home_km": 1.5,
    "route_cell_deg": 0.01,      # ~1.1 km grid for the learned corridor
    "route_min_support": 2,      # other sessions of same vehicle that used the cell
    "route_min_sessions": 6,     # vehicle needs history before we judge routes
    "route_min_minutes": 5.0,
    "fuel_drop_l": 1.0,
    "fuel_max_gap_s": 900.0,
    "corr_window_min": 15.0,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "speeding": 3.0,
    "harsh_driving": 2.0,
    "idling": 1.0,
    "route_deviation": 4.0,
    "unauthorized_stop": 4.0,
    "fuel_theft": 5.0,
    "fuel_off_route": 5.0,   # compound: fuel loss while off-route / at an unknown stop
}

BASE_PENALTY: Dict[str, float] = {
    "speeding": 1.0, "harsh_driving": 0.8, "idling": 0.4,
    "route_deviation": 1.5, "unauthorized_stop": 1.5,
    "fuel_theft": 2.0, "fuel_off_route": 3.0,
}

EVENT_COLUMNS = ["kind", "vehicle_id", "session_id", "start", "end",
                 "duration_s", "magnitude", "lat", "lon"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ preparation
def prepare(pings: pd.DataFrame) -> pd.DataFrame:
    """Split ``month_day_vehicle`` ids into vehicle + session and add deltas."""
    required = {"vehicle_id", "timestamp", "latitude", "longitude", "speed",
                "fuel_level", "ignition"}
    missing = required - set(pings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if pings.empty:
        raise ValueError("No telemetry rows supplied")

    df = pings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    df["session_id"] = df["vehicle_id"].astype(str)
    df["vehicle_id"] = df["session_id"].str.rsplit("_", n=1).str[-1]
    df["ignition"] = pd.to_numeric(df["ignition"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    g = df.groupby("session_id", sort=False)
    df["dt"] = g["timestamp"].diff().dt.total_seconds()
    df["dv"] = g["speed"].diff()
    df["dl"] = g["fuel_level"].diff()
    plat, plon = g["latitude"].shift(), g["longitude"].shift()
    df["step_km"] = haversine_km(plat, plon, df["latitude"], df["longitude"]).fillna(0.0)
    return df


def _cells(lat: pd.Series, lon: pd.Series, deg: float):
    return (np.floor(lat / deg).astype("int64"), np.floor(lon / deg).astype("int64"))


def _runs(df: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """Run id for consecutive flagged rows inside one session (NaN if not flagged)."""
    sess_change = df["session_id"] != df["session_id"].shift()
    new = flag & (~flag.shift(fill_value=False) | sess_change)
    run = new.cumsum().where(flag)
    return run


def _episodes(df: pd.DataFrame, flag: pd.Series, kind: str, mag: Optional[pd.Series] = None,
              min_duration_s: float = 0.0, mag_agg: str = "max") -> pd.DataFrame:
    run = _runs(df, flag)
    if run.notna().sum() == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    sub = df.loc[flag, ["vehicle_id", "session_id", "timestamp", "latitude", "longitude"]].copy()
    sub["run"] = run[flag].to_numpy()
    sub["mag"] = (mag[flag] if mag is not None else 0.0)
    ag = sub.groupby("run").agg(
        vehicle_id=("vehicle_id", "first"), session_id=("session_id", "first"),
        start=("timestamp", "min"), end=("timestamp", "max"),
        lat=("latitude", "first"), lon=("longitude", "first"),
        magnitude=("mag", mag_agg))
    ag["duration_s"] = (ag["end"] - ag["start"]).dt.total_seconds()
    ag = ag[ag["duration_s"] >= min_duration_s]
    ag["kind"] = kind
    return ag.reset_index(drop=True)[EVENT_COLUMNS]


def _neighbour_support(keys: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Distinct sessions covering each (group, cell) incl. its 8 neighbours."""
    u = keys[[group_col, "session_id", "cx", "cy"]].drop_duplicates()
    parts = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            p = u.copy()
            p["cx"] += dx
            p["cy"] += dy
            parts.append(p)
    big = pd.concat(parts).drop_duplicates()
    sup = big.groupby([group_col, "cx", "cy"])["session_id"].nunique().rename("support")
    return sup.reset_index()


# ---------------------------------------------------------------- home & places
def vehicle_home_bases(df: pd.DataFrame, deg: float = 0.005) -> pd.DataFrame:
    """Home base = cell where the vehicle spends the most stationary time."""
    st = df[(df["speed"] < 1.0) & df["dt"].notna() & (df["dt"] < 86400)].copy()
    st["cx"], st["cy"] = _cells(st["latitude"], st["longitude"], deg)
    t = st.groupby(["vehicle_id", "cx", "cy"]).agg(
        secs=("dt", "sum"), lat=("latitude", "median"), lon=("longitude", "median")).reset_index()
    t = t.sort_values("secs", ascending=False).drop_duplicates("vehicle_id")
    return t[["vehicle_id", "lat", "lon"]].rename(columns={"lat": "home_lat", "lon": "home_lon"})


# -------------------------------------------------------------------- detectors
def detect_speeding(df, p):
    flag = df["speed"] > p["speed_limit_kmh"]
    return _episodes(df, flag, "speeding", df["speed"], mag_agg="max")


def detect_harsh(df, p):
    ok = df["dt"].between(0.5, p["harsh_max_dt_s"])
    acc = (df["dv"] / 3.6) / df["dt"].where(ok)
    flag = acc.abs() >= p["harsh_mps2"]
    flag = flag.fillna(False)
    return _episodes(df, flag, "harsh_driving", acc.abs(), mag_agg="max")


def detect_idling(df, p):
    flag = (df["speed"] < 1.0) & (df["ignition"] == 1)
    return _episodes(df, flag, "idling", None, min_duration_s=p["idle_sec"])


def _stationary_episodes(df, min_s):
    return _episodes(df, df["speed"] < 1.0, "stop", None, min_duration_s=min_s)


def detect_unauthorized_stops(df, p, homes=None):
    homes = vehicle_home_bases(df) if homes is None else homes
    stops = _stationary_episodes(df, p["stop_min_minutes"] * 60.0)
    if stops.empty:
        return stops
    # known places: stationary spots (>=10 min) visited by many sessions fleet-wide
    known = _stationary_episodes(df, 600.0)
    known["cx"], known["cy"] = _cells(known["lat"], known["lon"], p["stop_cell_deg"])
    known["fleet"] = 1
    sup = _neighbour_support(known, "fleet")
    stops["cx"], stops["cy"] = _cells(stops["lat"], stops["lon"], p["stop_cell_deg"])
    stops = stops.merge(sup[["cx", "cy", "support"]], on=["cx", "cy"], how="left")
    stops["support"] = stops["support"].fillna(0)
    stops = stops.merge(homes, on="vehicle_id", how="left")
    d_home = haversine_km(stops["lat"], stops["lon"], stops["home_lat"], stops["home_lon"])
    # support includes the stop's own session -> require >= N distinct sessions
    unknown = (stops["support"] < p["stop_known_sessions"]) & (d_home > p["stop_home_km"])
    out = stops[unknown].copy()
    out["kind"] = "unauthorized_stop"
    out["magnitude"] = out["duration_s"] / 60.0     # minutes
    return out[EVENT_COLUMNS].reset_index(drop=True)


def detect_route_deviation(df, p):
    """Moving off the vehicle's own learned corridor (leave-one-session-out)."""
    d = df.copy()
    d["cx"], d["cy"] = _cells(d["latitude"], d["longitude"], p["route_cell_deg"])
    nsess = d.groupby("vehicle_id")["session_id"].nunique()
    eligible = nsess[nsess >= p["route_min_sessions"]].index
    moving = d[(d["speed"] >= 5.0) & d["vehicle_id"].isin(eligible)]
    if moving.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    sup = _neighbour_support(d[d["vehicle_id"].isin(eligible)], "vehicle_id")
    m = moving.merge(sup, on=["vehicle_id", "cx", "cy"], how="left")
    m["others"] = m["support"].fillna(1) - 1          # remove the session itself
    flag = pd.Series(False, index=df.index)
    flag.loc[moving.index] = (m["others"].to_numpy() < p["route_min_support"])
    return _episodes(df, flag, "route_deviation", df["step_km"],
                     min_duration_s=p["route_min_minutes"] * 60.0, mag_agg="sum")


def detect_fuel_theft(df, p):
    g = df.groupby("session_id", sort=False)
    prev_level = df["fuel_level"] - df["dl"]
    nxt = g["fuel_level"].shift(-1)
    cand = ((df["dl"] <= -p["fuel_drop_l"]) & (df["speed"] < 1.0)
            & (df["ignition"] == 0) & (df["dt"] <= p["fuel_max_gap_s"]))
    persists = nxt.isna() | (nxt < prev_level - 0.5 * p["fuel_drop_l"])
    flag = (cand & persists).fillna(False)
    return _episodes(df, flag, "fuel_theft", -df["dl"], mag_agg="sum")


def correlate_fuel_with_route(fuel: pd.DataFrame, others: pd.DataFrame, window_min: float):
    """Fuel loss within +-window of an off-route episode / unknown stop (same session)."""
    if fuel.empty or others.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    w = pd.Timedelta(minutes=window_min)
    m = fuel.merge(others[["session_id", "start", "end", "kind"]],
                   on="session_id", suffixes=("", "_ctx"))
    hit = (m["start"] <= m["end_ctx"] + w) & (m["end"] >= m["start_ctx"] - w)
    m = m[hit].drop_duplicates(subset=["session_id", "start"])
    out = m[EVENT_COLUMNS].copy()
    out["kind"] = "fuel_off_route"
    return out.reset_index(drop=True)


def detect_all(pings: pd.DataFrame, params: Optional[Dict[str, float]] = None):
    """Run the full pipeline. Returns (prepared_df, events_df)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    df = prepare(pings)
    homes = vehicle_home_bases(df)
    speeding = detect_speeding(df, p)
    harsh = detect_harsh(df, p)
    idle = detect_idling(df, p)
    stops = detect_unauthorized_stops(df, p, homes)
    route = detect_route_deviation(df, p)
    fuel = detect_fuel_theft(df, p)
    comp = correlate_fuel_with_route(fuel, pd.concat([route, stops], ignore_index=True),
                                     p["corr_window_min"])
    parts = [x for x in (speeding, harsh, idle, stops, route, fuel, comp) if not x.empty]
    events = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=EVENT_COLUMNS))
    events = events.sort_values("start").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    return df, events


# ---------------------------------------------------------------------- scoring
def event_severity(ev: pd.DataFrame, speed_limit: float = 90.0) -> pd.Series:
    k, m, d = ev["kind"], ev["magnitude"].astype(float), ev["duration_s"].astype(float)
    sev = pd.Series(1.0, index=ev.index)
    sev[k == "speeding"] = (1 + (m - speed_limit) / 10.0).clip(1, 3)
    sev[k == "idling"] = (d / 300.0).clip(1, 3)
    sev[k == "route_deviation"] = (d / 600.0).clip(1, 3)
    sev[k == "unauthorized_stop"] = (d / 1800.0).clip(1, 3)
    sev[k.isin(["fuel_theft", "fuel_off_route"])] = (m / 10.0).clip(1, 3)
    return sev


MIN_EXPOSURE_KM = 50.0


def exposure_100km(km) -> "pd.Series | float":
    """Exposure divisor for session scoring: distance driven in units of 100 km.

    Distances under ``MIN_EXPOSURE_KM`` are floored so one incident on a short or
    parked session is not blown up by dividing by almost nothing.
    """
    return np.maximum(km, MIN_EXPOSURE_KM) / 100.0


def tier(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 75: return "Low Risk"
    if score >= 50: return "Moderate Risk"
    if score >= 25: return "High Risk"
    return "Critical"


def dss(events: pd.DataFrame, as_of: Optional[pd.Timestamp] = None,
        weights: Optional[Dict[str, float]] = None, half_life_h: float = 6.0,
        r0: float = 10.0, group: str = "vehicle_id", universe=None,
        speed_limit_kmh: float = 90.0, exposure: Optional[pd.Series] = None) -> pd.DataFrame:
    """Driver Safety Score per vehicle (or session) as of ``as_of``.

    ``exposure`` (optional, indexed by group) divides the risk, so session scores
    become risk per 100 km driven and long and short trips are comparable.

    DSS = 100 * exp(-R / r0); R = sum w_k/3 * base_k * severity * decay.
    Smooth and bounded in (0, 100], so it never saturates at zero and recovers
    as events age (``half_life_h`` <= 0 disables decay).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    ev = events.copy()
    if as_of is not None:
        ev = ev[ev["end"] <= as_of]
    cols = list(DEFAULT_WEIGHTS)
    if ev.empty:
        idx = list(universe) if universe is not None else []
        return pd.DataFrame({group: idx, "score": 100.0, "tier": "Excellent", "risk": 0.0,
                             **{c: 0.0 for c in cols}})
    ev["sev"] = event_severity(ev, speed_limit_kmh)
    if as_of is not None and half_life_h and half_life_h > 0:
        age_h = (as_of - ev["end"]).dt.total_seconds() / 3600.0
        decay = 0.5 ** (age_h / half_life_h)
    else:
        decay = 1.0
    ev["pts"] = (ev["kind"].map(w) / 3.0) * ev["kind"].map(BASE_PENALTY) * ev["sev"] * decay
    piv = ev.pivot_table(index=group, columns="kind", values="pts", aggfunc="sum", fill_value=0.0)
    for c in cols:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[cols]
    if universe is not None:
        piv = piv.reindex(list(universe), fill_value=0.0)
    if exposure is not None:
        piv = piv.div(exposure.reindex(piv.index).fillna(1.0).to_numpy(), axis=0)
    piv["risk"] = piv[cols].sum(axis=1)
    piv["score"] = 100.0 * np.exp(-piv["risk"] / r0)
    piv["tier"] = piv["score"].map(tier)
    return piv.reset_index()


# ----------------------------------------------------------------------- replay
def fleet_snapshot(df: pd.DataFrame, t: pd.Timestamp, stale_min: float = 60.0) -> pd.DataFrame:
    """Last known position of every session active at simulated time ``t``."""
    past = df[df["timestamp"] <= t]
    last = past.groupby("session_id", sort=False).tail(1)
    last = last[(t - last["timestamp"]).dt.total_seconds() <= stale_min * 60]
    # a session is finished once its final ping is in the past (no future pings)
    end = df.groupby("session_id")["timestamp"].max()
    last = last[last["session_id"].map(end) >= t - pd.Timedelta(minutes=stale_min)]
    return last[["vehicle_id", "session_id", "timestamp", "latitude", "longitude",
                 "speed", "ignition", "fuel_level"]].reset_index(drop=True)
