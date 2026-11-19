"""Driver performance aggregation: ranking, weekly trend and behaviour mix."""
from __future__ import annotations

from datetime import date as date_type
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import dash_core as C
from src import risk_engine as R

DRIVING_KM = 50.0
MIN_TREND_DAYS = 2      # driving days a month needs before it counts for the trend


def driving(scored: pd.DataFrame, start: Optional[date_type] = None, end: Optional[date_type] = None) -> pd.DataFrame:
    """Sessions in range where the vehicle actually drove (>= DRIVING_KM)."""
    d = scored[(scored["km"] >= DRIVING_KM) & (scored["pings"] >= C.MIN_PINGS_REPLAY)].copy()
    if start is not None:
        d = d[d["date"] >= start]
    if end is not None:
        d = d[d["date"] <= end]
    return d


def weekly_trend(scored: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Mean session DSS per vehicle and per week (columns: week, vehicle_id, score, trips)."""
    d = driving(scored, start, end)
    if d.empty:
        return pd.DataFrame(columns=["week", "vehicle_id", "score", "trips"])
    d["week"] = pd.to_datetime(d["start"]).dt.to_period("W").dt.start_time
    return (d.groupby(["week", "vehicle_id"]).agg(score=("score", "mean"), trips=("score", "size"))
            .reset_index().sort_values(["vehicle_id", "week"]))


def fleet_weekly(trend: pd.DataFrame, scored_driving: pd.DataFrame) -> pd.DataFrame:
    if scored_driving.empty:
        return pd.DataFrame(columns=["week", "score"])
    d = scored_driving.copy()
    d["week"] = pd.to_datetime(d["start"]).dt.to_period("W").dt.start_time
    return d.groupby("week")["score"].mean().reset_index()


def driver_table(scored: pd.DataFrame, events: pd.DataFrame, weights: Dict[str, float], r0: float,
                 start=None, end=None, speed_limit: float = 90.0) -> pd.DataFrame:
    """One row per vehicle for the selected period."""
    d = driving(scored, start, end)
    cols = ["vehicle_id", "Driving days", "Distance (km)", "Mean DSS", "Lowest DSS", "Incidents / 100 km",
            "Main issue", "Trend (pts)", "Tier", "Data"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    g = d.groupby("vehicle_id")
    out = pd.DataFrame({
        "Driving days": g.size(),
        "Distance (km)": g["km"].sum(),
        "Mean DSS": g["score"].mean(),
        "Lowest DSS": g["score"].min(),
    })
    ev = events[events["session_id"].isin(d["session_id"])]
    inc = ev.groupby("vehicle_id").size().reindex(out.index).fillna(0)
    out["Incidents / 100 km"] = inc / (out["Distance (km)"] / 100.0)

    expo = pd.Series(R.exposure_100km(out["Distance (km)"].to_numpy()), index=out.index)
    pts = R.dss(ev, as_of=None, weights=weights, half_life_h=0, r0=r0, group="vehicle_id",
                universe=out.index.tolist(), speed_limit_kmh=speed_limit, exposure=expo)
    pts = pts.set_index("vehicle_id")
    kinds = list(R.DEFAULT_WEIGHTS)
    main = pts[kinds].idxmax(axis=1).where(pts[kinds].sum(axis=1) > 1e-9, "-")
    out["Main issue"] = main.map(lambda k: C.KIND_LABEL.get(k, "-")).reindex(out.index)

    d = d.assign(_month=pd.to_datetime(d["start"]).dt.to_period("M"))
    mon = d.groupby(["vehicle_id", "_month"]).agg(score=("score", "mean"), n=("score", "size")).reset_index()
    mon = mon[mon["n"] >= MIN_TREND_DAYS].sort_values(["vehicle_id", "_month"])
    last2 = mon.groupby("vehicle_id").tail(2)
    delta = last2.groupby("vehicle_id")["score"].agg(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 2 else np.nan)
    out["Trend (pts)"] = delta.reindex(out.index)
    out["Data"] = np.where(out["Driving days"] < 5, "Limited data", "")
    out["Tier"] = out["Mean DSS"].map(R.tier)
    out = out.reset_index().sort_values("Mean DSS").reset_index(drop=True)
    return out[cols]


def behaviour_mix(scored: pd.DataFrame, events: pd.DataFrame, weights: Dict[str, float], r0: float,
                  start=None, end=None, speed_limit: float = 90.0) -> pd.DataFrame:
    """Risk points per 100 km by behaviour for each vehicle (wide: vehicle_id + one column per kind)."""
    d = driving(scored, start, end)
    if d.empty:
        return pd.DataFrame(columns=["vehicle_id"] + list(R.DEFAULT_WEIGHTS))
    km = d.groupby("vehicle_id")["km"].sum()
    ev = events[events["session_id"].isin(d["session_id"])]
    pts = R.dss(ev, as_of=None, weights=weights, half_life_h=0, r0=r0, group="vehicle_id",
                universe=km.index.tolist(), speed_limit_kmh=speed_limit,
                exposure=pd.Series(R.exposure_100km(km.to_numpy()), index=km.index))
    return pts[["vehicle_id"] + list(R.DEFAULT_WEIGHTS)]
