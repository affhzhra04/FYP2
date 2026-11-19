"""Dashboard data layer (no Streamlit / Plotly imports, so it is unit-testable).

Turns the episode engine (``risk_engine``) into the objects the UI needs:
bundle loading, replay-day selection, live fleet state at a simulated time,
scoring tables, session audits and downloadable reports.
"""

from __future__ import annotations

import html
import io
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import risk_engine as R

KIND_LABEL: Dict[str, str] = {
    "speeding": "Speeding",
    "harsh_driving": "Harsh driving",
    "idling": "Excessive idling",
    "route_deviation": "Route deviation",
    "unauthorized_stop": "Unauthorized stop",
    "fuel_theft": "Fuel loss (stationary)",
    "fuel_off_route": "Fuel loss + off-route / unknown stop",
}
TIER_COLOR: Dict[str, str] = {
    "Excellent": "#16a34a",
    "Low Risk": "#84cc16",
    "Moderate Risk": "#f59e0b",
    "High Risk": "#f97316",
    "Critical": "#dc2626",
}
MIN_PINGS_REPLAY = 20


class DataUnavailable(Exception):
    """Raised with a user-friendly message when telemetry cannot be loaded."""


# ------------------------------------------------------------------- loading
def read_pings(db_path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise DataUnavailable(
            f"Database not found at {path}. Run the ingestion step first: "
            "python -m src.ingest --data-dir <folder with the JSON files>")
    try:
        with closing(sqlite3.connect(str(path))) as conn:
            df = pd.read_sql_query(
                "SELECT vehicle_id, timestamp, latitude, longitude, speed, fuel_level, "
                "heading, ignition FROM telematics_pings", conn)
    except Exception as exc:
        raise DataUnavailable(f"Could not read telematics_pings from {path.name}: {exc}") from exc
    if df.empty:
        raise DataUnavailable(
            "The telematics_pings table is empty. Run: python -m src.ingest --data-dir <folder>")
    return df


@dataclass
class Bundle:
    df: pd.DataFrame
    events: pd.DataFrame
    sessions: pd.DataFrame
    params: Dict[str, float] = field(default_factory=dict)


def session_base(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("session_id", sort=False)
    s = g.agg(vehicle_id=("vehicle_id", "first"), start=("timestamp", "min"),
              end=("timestamp", "max"), pings=("timestamp", "size"),
              km=("step_km", "sum"), max_speed=("speed", "max")).reset_index()
    s["date"] = s["start"].dt.date
    s["hours"] = (s["end"] - s["start"]).dt.total_seconds() / 3600.0
    return s.sort_values("start").reset_index(drop=True)


def build_bundle(pings: pd.DataFrame, params: Optional[Dict[str, float]] = None) -> Bundle:
    df, events = R.detect_all(pings, params)
    return Bundle(df=df, events=events, sessions=session_base(df),
                  params={**R.DEFAULT_PARAMS, **(params or {})})


# -------------------------------------------------------------------- helpers
def event_detail(row) -> str:
    k, m, d = row["kind"], float(row["magnitude"]), float(row["duration_s"])
    if k == "speeding":
        return f"peak {m:.0f} km/h"
    if k == "harsh_driving":
        return f"{m:.1f} m/s²"
    if k == "idling":
        return f"{d / 60:.0f} min idling"
    if k == "route_deviation":
        return f"{m:.1f} km off corridor, {d / 60:.0f} min"
    if k == "unauthorized_stop":
        return f"stopped {d / 60:.0f} min"
    if k in ("fuel_theft", "fuel_off_route"):
        return f"{m:.1f} L lost"
    return ""


def label_events(ev: pd.DataFrame) -> pd.DataFrame:
    out = ev.copy()
    out["event"] = out["kind"].map(KIND_LABEL)
    out["detail"] = out.apply(event_detail, axis=1) if len(out) else ""
    return out


def zoom_for(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    mid = np.radians((lat_min + lat_max) / 2.0)
    span = max(lat_max - lat_min, (lon_max - lon_min) * float(np.cos(mid)), 0.01) * 1.25
    return float(np.clip(np.log2(600.0 / span), 4.0, 14.0))


# ---------------------------------------------------------------- replay days
def replay_days(sessions: pd.DataFrame) -> pd.DataFrame:
    s = sessions[sessions["pings"] >= MIN_PINGS_REPLAY]
    d = s.groupby("date").agg(vehicles=("vehicle_id", "nunique"), sessions=("session_id", "nunique"),
                              pings=("pings", "sum")).reset_index()
    return d.sort_values("date").reset_index(drop=True)


def best_day(days: pd.DataFrame):
    if days.empty:
        return None
    return days.sort_values(["vehicles", "pings"], ascending=False).iloc[0]["date"]


@dataclass
class DayData:
    date: date_type
    session_ids: List[str]
    frame: pd.DataFrame
    vehicles: List[str]
    t0: pd.Timestamp
    t1: pd.Timestamp
    bbox: Dict[str, float]


def day_data(bundle: Bundle, day: date_type) -> DayData:
    s = bundle.sessions
    s = s[(s["date"] == day) & (s["pings"] >= MIN_PINGS_REPLAY)]
    if s.empty:
        raise DataUnavailable(f"No replayable sessions on {day}.")
    ids = s["session_id"].tolist()
    frame = bundle.df[bundle.df["session_id"].isin(ids)].sort_values("timestamp").reset_index(drop=True)
    return DayData(
        date=day, session_ids=ids, frame=frame,
        vehicles=sorted(s["vehicle_id"].unique().tolist(), key=lambda x: (len(x), x)),
        t0=frame["timestamp"].min().floor("h"), t1=frame["timestamp"].max().ceil("h"),
        bbox=dict(lat_min=float(frame["latitude"].min()), lat_max=float(frame["latitude"].max()),
                  lon_min=float(frame["longitude"].min()), lon_max=float(frame["longitude"].max())))


# --------------------------------------------------------------- live state
@dataclass
class LiveState:
    t: pd.Timestamp
    snapshot: pd.DataFrame
    leaderboard: pd.DataFrame
    trails: pd.DataFrame
    recent_events: pd.DataFrame
    feed: pd.DataFrame
    kpis: Dict[str, object]


def live_state(bundle: Bundle, day: DayData, t: pd.Timestamp, weights: Dict[str, float],
               half_life_h: float = 6.0, r0: float = 10.0, trail_min: float = 60.0,
               event_min: float = 90.0, feed_n: int = 12) -> LiveState:
    t = pd.Timestamp(t)
    snap = R.fleet_snapshot(day.frame, t)
    snap = snap.sort_values("timestamp").drop_duplicates("vehicle_id", keep="last")

    sc = R.dss(bundle.events, as_of=t, weights=weights, half_life_h=half_life_h, r0=r0,
               universe=day.vehicles, speed_limit_kmh=bundle.params["speed_limit_kmh"])
    kinds = list(R.DEFAULT_WEIGHTS)
    top = sc[kinds].idxmax(axis=1).where(sc["risk"] > 1e-9, "-").map(lambda k: KIND_LABEL.get(k, "-"))
    lb = sc[["vehicle_id", "score", "tier", "risk"]].copy()
    lb["top_factor"] = top.to_numpy()
    lb = lb.merge(snap[["vehicle_id", "speed", "fuel_level"]], on="vehicle_id", how="left")
    lb["status"] = np.where(lb["speed"].isna(), "Not reporting", np.where(lb["speed"] >= 5, "Moving", "Stopped"))
    lb = lb.sort_values("score").reset_index(drop=True)

    recent = day.frame[(day.frame["timestamp"] <= t) & (day.frame["timestamp"] > t - pd.Timedelta(minutes=trail_min))]
    trails = recent[["session_id", "vehicle_id", "timestamp", "latitude", "longitude"]]

    ev = label_events(bundle.events[bundle.events["session_id"].isin(day.session_ids)
                                     & (bundle.events["end"] <= t)])
    rec = ev[ev["end"] > t - pd.Timedelta(minutes=event_min)]
    feed = ev.sort_values("end", ascending=False).head(feed_n)[["end", "vehicle_id", "event", "detail"]]
    feed = feed.rename(columns={"end": "time", "vehicle_id": "vehicle"})

    kpis = {
        "active": int(len(snap)),
        "vehicles": len(day.vehicles),
        "fleet_mean": float(lb["score"].mean()) if len(lb) else 100.0,
        "worst_vehicle": str(lb.iloc[0]["vehicle_id"]) if len(lb) else "-",
        "worst_score": float(lb.iloc[0]["score"]) if len(lb) else 100.0,
        "events_total": int(len(ev)),
        "events_hour": int((ev["end"] > t - pd.Timedelta(hours=1)).sum()),
    }
    return LiveState(t=t, snapshot=snap, leaderboard=lb, trails=trails,
                     recent_events=rec, feed=feed, kpis=kpis)


# ------------------------------------------------------------ scoring tables
def _exposure(sessions: pd.DataFrame) -> pd.Series:
    """Session exposure in units of 100 km (see ``risk_engine.exposure_100km``)."""
    return pd.Series(R.exposure_100km(sessions["km"].to_numpy()), index=sessions["session_id"].to_numpy())


def session_scores(bundle: Bundle, weights: Dict[str, float], r0: float = 10.0) -> pd.DataFrame:
    s = bundle.sessions.copy()
    sc = R.dss(bundle.events, as_of=None, weights=weights, half_life_h=0, r0=r0,
               group="session_id", universe=s["session_id"].tolist(),
               speed_limit_kmh=bundle.params["speed_limit_kmh"], exposure=_exposure(bundle.sessions))
    s = s.merge(sc[["session_id", "score", "tier", "risk"]], on="session_id", how="left")
    cnt = bundle.events.groupby("session_id").size().rename("events")
    s = s.merge(cnt, on="session_id", how="left")
    s["events"] = s["events"].fillna(0).astype(int)
    s["score"] = s["score"].fillna(100.0)
    s["tier"] = s["tier"].fillna("Excellent")
    return s


def compare_scores(bundle: Bundle, day: date_type, weights: Dict[str, float], r0: float = 10.0) -> pd.DataFrame:
    s = bundle.sessions[(bundle.sessions["date"] == day) & (bundle.sessions["pings"] >= MIN_PINGS_REPLAY)]
    ids = s["session_id"].tolist()
    lim = bundle.params["speed_limit_kmh"]
    expo = _exposure(bundle.sessions)
    base = R.dss(bundle.events, weights=None, half_life_h=0, r0=r0, group="session_id", universe=ids,
                 speed_limit_kmh=lim, exposure=expo)
    cust = R.dss(bundle.events, weights=weights, half_life_h=0, r0=r0, group="session_id", universe=ids,
                 speed_limit_kmh=lim, exposure=expo)
    out = s[["session_id", "vehicle_id"]].merge(base[["session_id", "score"]].rename(columns={"score": "default"}),
                                                on="session_id")
    out = out.merge(cust[["session_id", "score"]].rename(columns={"score": "custom"}), on="session_id")
    out["delta"] = out["custom"] - out["default"]
    out["rank_default"] = out["default"].rank(method="min").astype(int)
    out["rank_custom"] = out["custom"].rank(method="min").astype(int)
    return out.sort_values("custom").reset_index(drop=True)


@dataclass
class SessionAudit:
    session_id: str
    meta: Dict[str, object]
    items: pd.DataFrame
    by_kind: pd.DataFrame
    risk: float
    score: float
    tier: str


def session_audit(bundle: Bundle, sid: str, weights: Dict[str, float], r0: float = 10.0) -> SessionAudit:
    w = {**R.DEFAULT_WEIGHTS, **(weights or {})}
    srow = bundle.sessions[bundle.sessions["session_id"] == sid]
    if srow.empty:
        raise KeyError(f"Unknown session {sid}")
    srow = srow.iloc[0]
    ev = bundle.events[bundle.events["session_id"] == sid].copy().sort_values("start")
    if ev.empty:
        items = pd.DataFrame(columns=["Time", "Event", "Detail", "Severity", "Weight", "Points", "kind"])
    else:
        sev = R.event_severity(ev, bundle.params["speed_limit_kmh"])
        ev = label_events(ev)
        ev["Severity"] = sev.to_numpy()
        ev["Weight"] = ev["kind"].map(w).to_numpy()
        ev["Points"] = (ev["Weight"] / 3.0) * ev["kind"].map(R.BASE_PENALTY) * ev["Severity"]
        items = ev.rename(columns={"start": "Time", "event": "Event", "detail": "Detail"})[
            ["Time", "Event", "Detail", "Severity", "Weight", "Points", "kind"]]
    raw = float(items["Points"].sum()) if len(items) else 0.0
    expo = float(R.exposure_100km(float(srow["km"])))
    risk = raw / expo                      # risk points per 100 km driven
    score = float(100.0 * np.exp(-risk / r0))
    by_kind = (items.groupby("kind")["Points"].sum().reindex(list(R.DEFAULT_WEIGHTS), fill_value=0.0)
               .rename("points").reset_index())
    by_kind["points"] = by_kind["points"] / expo
    by_kind["label"] = by_kind["kind"].map(KIND_LABEL)
    meta = dict(vehicle=srow["vehicle_id"], start=srow["start"], end=srow["end"], pings=int(srow["pings"]),
                km=float(srow["km"]), max_speed=float(srow["max_speed"]), hours=float(srow["hours"]),
                raw_risk=raw, exposure=expo)
    return SessionAudit(sid, meta, items.reset_index(drop=True), by_kind, risk, score, R.tier(score))


# ------------------------------------------------------------------- reports
def _fmt_ts(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def report_html(a: SessionAudit, weights: Dict[str, float], r0: float, params: Dict[str, float]) -> str:
    rows = "".join(
        f"<tr><td>{_fmt_ts(r.Time)}</td><td>{html.escape(str(r.Event))}</td><td>{html.escape(str(r.Detail))}</td>"
        f"<td>{r.Severity:.2f}</td><td>{r.Weight:.1f}</td><td>{r.Points:.2f}</td></tr>"
        for r in a.items.itertuples()) or "<tr><td colspan='6'>No incidents detected.</td></tr>"
    wrows = "".join(f"<tr><td>{KIND_LABEL[k]}</td><td>{weights.get(k, R.DEFAULT_WEIGHTS[k]):.1f}</td></tr>"
                    for k in R.DEFAULT_WEIGHTS)
    prows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in params.items())
    m = a.meta
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Driver risk report {html.escape(a.session_id)}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:32px;color:#0f172a}}h1{{margin:0}}
table{{border-collapse:collapse;margin:12px 0;width:100%}}td,th{{border:1px solid #cbd5e1;padding:6px 8px;font-size:13px;text-align:left}}
th{{background:#f1f5f9}}.score{{font-size:40px;font-weight:700}}.muted{{color:#64748b;font-size:13px}}
@media print{{body{{margin:12mm}}}}</style></head><body>
<h1>Driver Safety Report</h1><div class="muted">Session {html.escape(a.session_id)} · Vehicle {html.escape(str(m['vehicle']))}</div>
<p class="score">{a.score:.1f} / 100 <span class="muted">({a.tier})</span></p>
<table><tr><th>Start</th><th>End</th><th>Distance</th><th>Pings</th><th>Max speed</th></tr>
<tr><td>{_fmt_ts(m['start'])}</td><td>{_fmt_ts(m['end'])}</td><td>{m['km']:.1f} km</td><td>{m['pings']}</td><td>{m['max_speed']:.0f} km/h</td></tr></table>
<h3>Incidents and score deductions</h3>
<table><tr><th>Time</th><th>Event</th><th>Detail</th><th>Severity</th><th>Weight</th><th>Points</th></tr>{rows}</table>
<p>Raw risk points = {m['raw_risk']:.2f}; distance exposure = {m['exposure']:.2f} (units of 100 km, minimum 0.5).
R = {m['raw_risk']:.2f} / {m['exposure']:.2f} = {a.risk:.2f} per 100 km; DSS = 100 × exp(−R / {r0:g}) = <b>{a.score:.1f}</b>.</p>
<h3>Scoring weights used</h3><table><tr><th>Factor</th><th>Weight</th></tr>{wrows}</table>
<h3>Detection thresholds</h3><table><tr><th>Parameter</th><th>Value</th></tr>{prows}</table>
<p class="muted">Times are device timestamps as stored. Generated by the Driver Risk Monitor.</p></body></html>"""


def report_pdf(a: SessionAudit, weights: Dict[str, float], r0: float) -> Optional[bytes]:
    """PDF bytes, or None when reportlab is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    st_ = getSampleStyleSheet()
    m = a.meta
    story = [Paragraph("Driver Safety Report", st_["Title"]),
             Paragraph(f"Session {a.session_id} - Vehicle {m['vehicle']}", st_["Normal"]),
             Spacer(1, 6), Paragraph(f"<b>DSS {a.score:.1f} / 100</b> ({a.tier})", st_["Heading2"]),
             Paragraph(f"{_fmt_ts(m['start'])} to {_fmt_ts(m['end'])} - {m['km']:.1f} km - "
                       f"max {m['max_speed']:.0f} km/h", st_["Normal"]), Spacer(1, 8),
             Paragraph("Incidents and score deductions", st_["Heading3"])]
    data = [["Time", "Event", "Detail", "Sev.", "Wt", "Pts"]]
    for r in a.items.head(60).itertuples():
        data.append([_fmt_ts(r.Time), str(r.Event), str(r.Detail), f"{r.Severity:.2f}",
                     f"{r.Weight:.1f}", f"{r.Points:.2f}"])
    if len(data) == 1:
        data.append(["-", "No incidents detected", "", "", "", ""])
    tbl = Table(data, repeatRows=1, colWidths=[38 * mm, 52 * mm, 48 * mm, 12 * mm, 10 * mm, 12 * mm])
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                             ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [tbl, Spacer(1, 8),
              Paragraph(f"R = {m['raw_risk']:.2f} / {m['exposure']:.2f} (100 km) = {a.risk:.2f}; DSS = 100 x exp(-R / {r0:g}) = {a.score:.1f}", st_["Normal"])]
    if len(a.items) > 60:
        story.append(Paragraph(f"(+{len(a.items) - 60} more incidents not shown)", st_["Normal"]))
    doc.build(story)
    return buf.getvalue()
