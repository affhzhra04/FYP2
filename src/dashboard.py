"""Driver Risk Monitor - real-time multi-driver risk assessment dashboard.

Run from the project root:   streamlit run src/dashboard.py

Pages
  1. Live Monitor       - map, live driver safety scores and incident feed
  2. Incidents & Review - priority queue, review decisions with notes, supervisor alerts, unusual trips
  3. Driver Performance - ranking, weekly trend and behaviour mix per driver
  4. Scoring Policy     - decide how much each behaviour counts; preview the effect
  5. History & Reports  - trip history, filters, unusual-pattern flags, CSV / HTML / PDF reports
  6. System & Models    - data coverage and quality, risk-model versions with retraining, alert gateway
The model-evaluation view for the project panel is a separate app: streamlit run src/evaluation.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from database.db_manager import DEFAULT_DB_PATH
from src import dash_core as C
from src import risk_engine as R
from src import ml_layer as ML
from src import alerts as A
from src import performance as P
from src import review_store as RS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("driver_risk_dashboard")

st.set_page_config(page_title="Driver Risk Monitor", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """<style>
    .block-container{padding-top:1.6rem;max-width:1500px}
    [data-testid="stMetricValue"]{font-size:1.55rem}
    [data-testid="stMetricLabel"]{font-size:.8rem;color:#64748b}
    h1{font-size:1.7rem !important;margin-bottom:.1rem}
    </style>""",
    unsafe_allow_html=True,
)

PAGES = ["Live Monitor", "Incidents & Review", "Driver Performance", "Scoring Policy", "History & Reports",
         "System & Models"]
SPEEDS = {"1 min / s": 60, "5 min / s": 300, "15 min / s": 900, "1 h / s": 3600}
TICK_SLEEP_S = 0.5
VEHICLE_FOCUS_ALL = "All vehicles"
KIND_COLOR = {
    "speeding": "#ef4444", "harsh_driving": "#f97316", "idling": "#a3a3a3",
    "route_deviation": "#8b5cf6", "unauthorized_stop": "#0ea5e9",
    "fuel_theft": "#d97706", "fuel_off_route": "#be123c",
}


# ------------------------------------------------------------------ helpers
def show_fig(fig, key=None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:                      # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:                      # older Streamlit
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def fmt_ts(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def _map_trace():
    return go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox


def _map_layout(center_lat, center_lon, zoom, revision, height=520) -> dict:
    key = "map" if hasattr(go, "Scattermap") else "mapbox"
    return {
        key: dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        "margin": dict(l=0, r=0, t=0, b=0), "height": height, "uirevision": revision,
        "legend": dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                       bgcolor="rgba(255,255,255,0.85)"),
    }


# ------------------------------------------------------------------- sidebar
def _reset_formula() -> None:
    for k, v in R.DEFAULT_WEIGHTS.items():
        st.session_state[f"w_{k}"] = float(v)
    st.session_state["half_life"] = 6
    st.session_state["r0"] = 10.0


def sidebar():
    with st.sidebar:
        st.markdown("### Driver Risk Monitor")
        st.caption("Real-time multi-driver safety scoring")
        page = st.radio("View", PAGES, label_visibility="collapsed", key="page")
        st.divider()

        st.markdown("**Scoring policy**")
        st.caption("Raise a weight to make that behaviour cost more points.")
        weights = {}
        for k, default in R.DEFAULT_WEIGHTS.items():
            weights[k] = st.slider(C.KIND_LABEL[k], 0.0, 5.0, float(default), 0.5, key=f"w_{k}")
        half_life = st.slider("Score memory (hours)", 1, 72, 6, key="half_life",
                              help="Live Monitor: how long an incident keeps lowering a driver's score. Shorter = scores recover faster.")
        with st.expander("Strictness"):
            r0 = st.slider("Score strictness", 3.0, 30.0, 10.0, 1.0, key="r0",
                           help="Lower = stricter (the same incidents cost more points). Higher = more forgiving.")
        st.button("Reset formula", on_click=_reset_formula)

        st.divider()
        with st.expander("Detection thresholds"):
            p_speed = st.number_input("Speed limit (km/h)", 40, 140, 90, 5, key="p_speed")
            p_idle = st.number_input("Idling after (min)", 1, 60, 5, 1, key="p_idle")
            p_stop = st.number_input("Unauthorized stop after (min)", 5, 240, 30, 5, key="p_stop")
            p_fuel = st.number_input("Fuel-loss threshold (L)", 0.3, 20.0, 1.0, 0.1, key="p_fuel")
        st.caption("All times are Malaysia time (MYT).")

    params = {"speed_limit_kmh": float(p_speed), "idle_sec": float(p_idle) * 60.0,
              "stop_min_minutes": float(p_stop), "fuel_drop_l": float(p_fuel)}
    return page, weights, float(half_life), float(r0), params


@st.cache_resource(show_spinner="Loading telemetry and running detectors...")
def get_bundle(db_path: str, mtime: float, params_key: tuple):
    return C.build_bundle(C.read_pings(db_path), dict(params_key))


def get_day(bundle, day):
    cached = st.session_state.get("_day")
    if cached is not None and cached[0] == day and cached[1] is bundle:
        return cached[2]
    dd = C.day_data(bundle, day)
    st.session_state["_day"] = (day, bundle, dd)
    return dd


# ------------------------------------------------------------- page 1: live
def _toggle_play() -> None:
    st.session_state["play"] = not st.session_state.get("play", False)


def _reset_clock() -> None:
    day = st.session_state.get("_day")
    if day is not None:
        st.session_state["t"] = day[2].t0.to_pydatetime()
    st.session_state["play"] = False


def build_live_map(ls: C.LiveState, day: C.DayData, focus: str, revision: str):
    Trace = _map_trace()
    fig = go.Figure()
    for _, g in ls.trails.groupby("session_id"):
        fig.add_trace(Trace(lat=g["latitude"], lon=g["longitude"], mode="lines",
                            line=dict(width=3, color="#94a3b8"), hoverinfo="skip", showlegend=False))
    if len(ls.recent_events):
        e = ls.recent_events
        fig.add_trace(Trace(
            lat=e["lat"], lon=e["lon"], mode="markers", name="Incident (last 90 min)",
            marker=dict(size=13, color="#dc2626"),
            hovertext=("Vehicle " + e["vehicle_id"].astype(str) + "<br>" + e["event"] + "<br>" + e["detail"]),
            hovertemplate="%{hovertext}<extra></extra>"))
    veh = ls.snapshot.merge(ls.leaderboard[["vehicle_id", "score", "tier"]], on="vehicle_id", how="left")
    for tier_name, g in veh.groupby("tier"):
        fig.add_trace(Trace(
            lat=g["latitude"], lon=g["longitude"], mode="markers+text", name=tier_name,
            marker=dict(size=18, color=C.TIER_COLOR.get(tier_name, "#2563eb")),
            text=g["vehicle_id"].astype(str), textposition="top center",
            hovertext=("<b>Vehicle " + g["vehicle_id"].astype(str) + "</b><br>DSS " + g["score"].round(0).astype(int).astype(str)
                       + " (" + g["tier"].astype(str) + ")<br>" + g["speed"].round(0).astype(int).astype(str)
                       + " km/h - fuel " + g["fuel_level"].round(0).astype(int).astype(str) + " L"),
            hovertemplate="%{hovertext}<extra></extra>"))
    b = day.bbox
    center_lat, center_lon = (b["lat_min"] + b["lat_max"]) / 2, (b["lon_min"] + b["lon_max"]) / 2
    zoom = C.zoom_for(b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"])
    if focus != VEHICLE_FOCUS_ALL:
        row = ls.snapshot[ls.snapshot["vehicle_id"] == focus]
        if len(row):
            center_lat, center_lon, zoom = float(row.iloc[0]["latitude"]), float(row.iloc[0]["longitude"]), 11.0
    fig.update_layout(**_map_layout(center_lat, center_lon, zoom, revision))
    return fig


def focus_strip(day: C.DayData, vehicle: str, t: pd.Timestamp):
    f = day.frame[(day.frame["vehicle_id"] == vehicle) & (day.frame["timestamp"] <= t)
                  & (day.frame["timestamp"] > t - pd.Timedelta(hours=3))]
    if f.empty:
        st.caption(f"Vehicle {vehicle} has no pings in the last 3 hours of replay time.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Speed (km/h)", "Fuel level (L)"))
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["speed"], mode="lines", line=dict(color="#2563eb", width=2),
                             name="Speed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["fuel_level"], mode="lines", line=dict(color="#059669", width=2),
                             name="Fuel"), row=2, col=1)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      template="plotly_white")
    show_fig(fig, key="focus_strip")


def page_live(bundle: C.Bundle, weights, half_life, r0) -> None:
    st.title("Live Monitor")
    st.caption("Fleet positions and driver safety scores, updated as incidents occur. Recorded telemetry is replayed as a live feed.")

    days = C.replay_days(bundle.sessions)
    if days.empty:
        st.info("No sessions with enough telemetry to replay.")
        return
    options = days["date"].tolist()
    labels = {r.date: f"{pd.Timestamp(r.date):%d %b %Y} - {r.vehicles} vehicles" for r in days.itertuples()}
    if st.session_state.get("replay_date") not in options:
        st.session_state["replay_date"] = C.best_day(days)

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.7, 1.2, 1.4, 0.8, 0.8])
        sel = c1.selectbox("Replay day", options, key="replay_date", format_func=lambda d: labels[d])
        speed_lbl = c2.selectbox("Playback speed", list(SPEEDS), index=1, key="speed")
        day = get_day(bundle, sel)
        focus = c3.selectbox("Map focus", [VEHICLE_FOCUS_ALL] + day.vehicles, key="focus")
        playing = bool(st.session_state.get("play", False))
        c4.write("")
        c4.button("Pause" if playing else "Play", on_click=_toggle_play, type="primary", key="btn_play")
        c5.write("")
        c5.button("Restart", on_click=_reset_clock, key="btn_reset")

    t0, t1 = day.t0.to_pydatetime(), day.t1.to_pydatetime()
    if st.session_state.get("_last_date") != sel:
        st.session_state["_last_date"] = sel
        st.session_state["t"] = t0
        st.session_state["play"] = False
    cur = st.session_state.get("t")
    if cur is None or cur < t0 or cur > t1:
        st.session_state["t"] = t0

    if st.session_state.pop("_tick", False) and st.session_state.get("play"):
        nxt = st.session_state["t"] + timedelta(seconds=SPEEDS[speed_lbl])
        if nxt >= t1:
            nxt = t1
            st.session_state["play"] = False
        st.session_state["t"] = nxt

    t_sel = st.slider("Simulated time (device clock)", min_value=t0, max_value=t1,
                      step=timedelta(minutes=1), format="DD MMM HH:mm", key="t")
    t = pd.Timestamp(t_sel)
    ls = C.live_state(bundle, day, t, weights, half_life, r0)

    k = ls.kpis
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Simulated time", t.strftime("%d %b %H:%M"))
    m2.metric("Vehicles reporting", f"{k['active']} / {k['vehicles']}")
    m3.metric("Fleet mean DSS", f"{k['fleet_mean']:.0f}")
    m4.metric("Lowest DSS", f"{k['worst_score']:.0f}", delta=f"vehicle {k['worst_vehicle']}", delta_color="off")
    m5.metric("Incidents (last hour)", k["events_hour"], delta=f"{k['events_total']} today", delta_color="off")

    left, right = st.columns([2.1, 1.0], gap="medium")
    with left:
        show_fig(build_live_map(ls, day, focus, f"{sel}-{focus}"), key="live_map")
        st.caption("Marker colour = current risk tier. Grey line = last 60 min of route. Red dots = recent incidents.")
    with right:
        st.markdown("**Driver safety score (live)**")
        lb = ls.leaderboard.rename(columns={"vehicle_id": "Vehicle", "score": "DSS", "tier": "Tier",
                                             "status": "Status", "top_factor": "Main factor"})
        show_df(lb[["Vehicle", "DSS", "Tier", "Status", "Main factor"]], height=390,
                column_config={"DSS": st.column_config.ProgressColumn("DSS", min_value=0, max_value=100,
                                                                      format="%.0f")})
        st.markdown("**Incident feed**")
        if ls.feed.empty:
            st.caption("No incidents yet at this point in the replay.")
        else:
            feed = ls.feed.copy()
            feed["time"] = feed["time"].map(lambda x: pd.Timestamp(x).strftime("%H:%M"))
            show_df(feed, height=300)

    if focus != VEHICLE_FOCUS_ALL:
        st.markdown(f"**Vehicle {focus} - telemetry (last 3 h)**")
        focus_strip(day, focus, t)

    if st.session_state.get("play"):
        time.sleep(TICK_SLEEP_S)
        st.session_state["_tick"] = True
        st.rerun()


# ----------------------------------------------------------- page 2: scoring
def page_lab(bundle: C.Bundle, weights, half_life, r0) -> None:
    st.title("Scoring Policy")
    st.caption("Decide how much each behaviour counts towards the Driver Safety Score using the sliders on the left. "
               "The preview below shows how that would change this day's driver ranking.")

    st.subheader("Preview of your settings")
    days = C.replay_days(bundle.sessions)
    if days.empty:
        st.info("No sessions available.")
        return
    options = days["date"].tolist()
    labels = {r.date: f"{pd.Timestamp(r.date):%d %b %Y} - {r.vehicles} vehicles" for r in days.itertuples()}
    if st.session_state.get("lab_date") not in options:
        st.session_state["lab_date"] = C.best_day(days)
    day = st.selectbox("Compare on day", options, key="lab_date", format_func=lambda d: labels[d])
    cmp = C.compare_scores(bundle, day, weights, r0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cmp["vehicle_id"], y=cmp["default"], name="Default weights", marker_color="#94a3b8"))
    fig.add_trace(go.Bar(x=cmp["vehicle_id"], y=cmp["custom"], name="Your weights", marker_color="#2563eb"))
    fig.update_layout(barmode="group", height=320, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Vehicle ID", yaxis_title="Session DSS", yaxis_range=[0, 100],
                      legend=dict(orientation="h", y=1.1))
    show_fig(fig, key="lab_compare")
    tbl = cmp.rename(columns={"vehicle_id": "Vehicle ID", "default": "DSS (default)", "custom": "DSS (yours)",
                              "delta": "Change", "rank_default": "Risk rank (default)",
                              "rank_custom": "Risk rank (yours)"})
    show_df(tbl[["Vehicle ID", "DSS (default)", "DSS (yours)", "Change", "Risk rank (default)",
                 "Risk rank (yours)"]].round(1))
    st.caption("Risk rank 1 = highest risk. Vehicles with the same score share a rank.")

    st.caption("To see how one trip was scored, or to export a report, open History & Reports.")


def render_audit(bundle, sid, weights, r0, key_prefix: str) -> None:
    a = C.session_audit(bundle, sid, weights, r0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session DSS", f"{a.score:.1f}")
    c2.metric("Tier", a.tier)
    c3.metric("Risk per 100 km", f"{a.risk:.1f}")
    c4.metric("Incidents", len(a.items))
    if a.items.empty:
        st.success("No incidents detected in this session.")
        return
    bk = a.by_kind[a.by_kind["points"] > 0].sort_values("points")
    fig = go.Figure(go.Bar(x=bk["points"], y=bk["label"], orientation="h",
                           marker_color=[KIND_COLOR[k] for k in bk["kind"]]))
    fig.update_layout(height=max(180, 55 * len(bk)), template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Risk points per 100 km")
    show_fig(fig, key=f"{key_prefix}_bykind")
    items = a.items.drop(columns=["kind"]).copy()
    items["Time"] = items["Time"].map(fmt_ts)
    show_df(items.round(2), height=min(420, 40 + 35 * len(items)))
    m = a.meta
    st.info(f"{m['raw_risk']:.1f} risk points over {m['km']:.0f} km = {a.risk:.1f} per 100 km  ->  "
            f"Driver Safety Score **{a.score:.1f}** ({a.tier})")


# ------------------------------------------------------ page 3: sessions
def route_figure(bundle: C.Bundle, sid: str):
    f = bundle.df[bundle.df["session_id"] == sid]
    ev = C.label_events(bundle.events[bundle.events["session_id"] == sid])
    Trace = _map_trace()
    fig = go.Figure()
    fig.add_trace(Trace(lat=f["latitude"], lon=f["longitude"], mode="lines", line=dict(width=3, color="#2563eb"),
                        name="Route", hoverinfo="skip"))
    if len(f):
        fig.add_trace(Trace(lat=[f.iloc[0]["latitude"]], lon=[f.iloc[0]["longitude"]], mode="markers",
                            marker=dict(size=14, color="#16a34a"), name="Start"))
        fig.add_trace(Trace(lat=[f.iloc[-1]["latitude"]], lon=[f.iloc[-1]["longitude"]], mode="markers",
                            marker=dict(size=14, color="#111827"), name="End"))
    for kind, g in ev.groupby("kind"):
        fig.add_trace(Trace(lat=g["lat"], lon=g["lon"], mode="markers", name=C.KIND_LABEL[kind],
                            marker=dict(size=12, color=KIND_COLOR[kind]),
                            hovertext=g["event"] + "<br>" + g["detail"], hovertemplate="%{hovertext}<extra></extra>"))
    if len(f):
        lat_min, lat_max = float(f["latitude"].min()), float(f["latitude"].max())
        lon_min, lon_max = float(f["longitude"].min()), float(f["longitude"].max())
        fig.update_layout(**_map_layout((lat_min + lat_max) / 2, (lon_min + lon_max) / 2,
                                        C.zoom_for(lat_min, lat_max, lon_min, lon_max), sid, height=430))
    return fig


def timeseries_figure(bundle: C.Bundle, sid: str, speed_limit: float):
    f = bundle.df[bundle.df["session_id"] == sid]
    ev = C.label_events(bundle.events[bundle.events["session_id"] == sid])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Speed (km/h)", "Fuel level (L)"))
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["speed"], mode="lines", line=dict(color="#2563eb", width=1.6),
                             name="Speed"), row=1, col=1)
    fig.add_hline(y=speed_limit, line_dash="dash", line_color="#dc2626", row=1, col=1)
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["fuel_level"], mode="lines", line=dict(color="#059669", width=1.6),
                             name="Fuel"), row=2, col=1)
    if len(ev) and len(f):
        ts = f["timestamp"].to_numpy()
        for kind, g in ev.groupby("kind"):
            idx = np.clip(np.searchsorted(ts, g["start"].to_numpy()), 0, len(f) - 1)
            fig.add_trace(go.Scatter(x=g["start"], y=f["speed"].to_numpy()[idx], mode="markers", name=C.KIND_LABEL[kind],
                                     marker=dict(size=10, color=KIND_COLOR[kind]),
                                     hovertext=g["event"] + "<br>" + g["detail"],
                                     hovertemplate="%{hovertext}<extra></extra>"), row=1, col=1)
    fig.update_layout(height=420, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def page_sessions(bundle: C.Bundle, weights, half_life, r0) -> None:
    st.title("History & Reports")
    st.caption("One row per vehicle per day. Filter the history, open a trip to see its route and score, export a report.")

    ss = C.session_scores(bundle, weights, r0)
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 1.3, 1.4])
        vehicles = sorted(ss["vehicle_id"].unique().tolist(), key=lambda x: (len(x), x))
        sel_v = c1.multiselect("Vehicles", vehicles, default=vehicles, key="f_vehicles")
        dmin, dmax = ss["date"].min(), ss["date"].max()
        rng = c2.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="f_dates")
        tiers = ["Critical", "High Risk", "Moderate Risk", "Low Risk", "Excellent"]
        sel_t = c3.multiselect("Risk tier", tiers, default=tiers, key="f_tiers")
        hide_small = st.checkbox(f"Hide very short sessions (< {C.MIN_PINGS_REPLAY} pings)", value=True, key="f_small")

    lo, hi = (rng[0], rng[-1]) if isinstance(rng, (tuple, list)) and len(rng) else (dmin, dmax)
    f = ss[ss["vehicle_id"].isin(sel_v) & ss["tier"].isin(sel_t) & (ss["date"] >= lo) & (ss["date"] <= hi)]
    if hide_small:
        f = f[f["pings"] >= C.MIN_PINGS_REPLAY]
    if f.empty:
        st.info("No sessions match these filters.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sessions", len(f))
    m2.metric("Mean DSS", f"{f['score'].mean():.0f}")
    m3.metric("Critical / High", int(f["tier"].isin(["Critical", "High Risk"]).sum()))
    m4.metric("Incidents", int(f["events"].sum()))

    table = f.sort_values("score")[["session_id", "vehicle_id", "start", "end", "hours", "km", "max_speed",
                                    "events", "score", "tier"]].copy()
    try:
        ml = get_ml(bundle, tuple(sorted(bundle.params.items())))
        flagged = set(ml.table.loc[ml.table["anomaly"] >= 90, "session_id"])
        reasons = ml.table.set_index("session_id")["reason"]
        table["unusual"] = [reasons.get(x, "") if x in flagged else "" for x in table["session_id"]]
    except Exception as exc:                # scikit-learn missing or too little data: skip the flag
        logger.warning("Unusual-pattern flag skipped: %s", exc)
        table["unusual"] = ""
    table["start"] = table["start"].map(fmt_ts)
    table["end"] = table["end"].map(fmt_ts)
    table = table.rename(columns={"session_id": "Session", "vehicle_id": "Vehicle ID", "start": "Start", "end": "End",
                                  "hours": "Hours", "km": "km", "max_speed": "Max km/h", "events": "Incidents",
                                  "score": "DSS", "tier": "Tier", "unusual": "Unusual pattern"}).round(1)
    if (table["Unusual pattern"] == "").all():
        table = table.drop(columns=["Unusual pattern"])
    show_df(table, height=360, column_config={"DSS": st.column_config.ProgressColumn(
        "DSS", min_value=0, max_value=100, format="%.0f")})

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download sessions (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="sessions.csv", mime="text/csv", key="dl_sessions")
    ev_f = C.label_events(bundle.events[bundle.events["session_id"].isin(f["session_id"])])
    ev_csv = ev_f[["session_id", "vehicle_id", "event", "start", "end", "duration_s", "magnitude", "detail"]]
    d2.download_button("Download incidents (CSV)", ev_csv.to_csv(index=False).encode("utf-8"),
                       file_name="incidents.csv", mime="text/csv", key="dl_events")

    st.subheader("Session detail")
    f_sorted = f.sort_values("score")
    lab = {r.session_id: f"{r.session_id} - vehicle {r.vehicle_id} - DSS {r.score:.0f} ({r.events} incidents)"
           for r in f_sorted.itertuples()}
    sid = st.selectbox("Session", f_sorted["session_id"].tolist(), format_func=lambda s: lab[s], key="detail_session")
    t1, t2 = st.tabs(["Route and telemetry", "Score audit and report"])
    with t1:
        show_fig(route_figure(bundle, sid), key="detail_route")
        show_fig(timeseries_figure(bundle, sid, bundle.params["speed_limit_kmh"]), key="detail_ts")
    with t2:
        render_audit(bundle, sid, weights, r0, key_prefix="det")
        a = C.session_audit(bundle, sid, weights, r0)
        r1, r2, _ = st.columns([1, 1, 2])
        r1.download_button("Download report (HTML)", C.report_html(a, weights, r0, bundle.params).encode("utf-8"),
                           file_name=f"report_{sid}.html", mime="text/html", key="dl_html")
        pdf = C.report_pdf(a, weights, r0)
        if pdf is not None:
            r2.download_button("Download report (PDF)", pdf, file_name=f"report_{sid}.pdf",
                               mime="application/pdf", key="dl_pdf")
        else:
            r2.caption("Install reportlab (pip install reportlab) to enable PDF export.")


# --------------------------------------------------------- incidents & review
def _review_form(item_id: str, item_type: str, current: dict, key: str) -> None:
    with st.form(key=f"form_{key}", clear_on_submit=False):
        status = st.selectbox("Decision", RS.STATUSES, key=f"st_{key}",
                              index=RS.STATUSES.index(current["status"]) if current.get("status") in RS.STATUSES else 0)
        reviewer = st.text_input("Reviewed by", value=st.session_state.get("reviewer", current.get("reviewer", "")),
                                 key=f"rv_{key}")
        note = st.text_area("Note (what did you find?)", value=current.get("note", ""), key=f"nt_{key}", height=90)
        if st.form_submit_button("Save review"):
            RS.save_review(item_id, item_type, status, note, reviewer)
            st.session_state["reviewer"] = reviewer
            st.success("Review saved.")
            st.rerun()


def _incident_frame(bundle: C.Bundle) -> pd.DataFrame:
    ev = C.label_events(bundle.events)
    ev["priority"] = A.priority(bundle.events, bundle.params["speed_limit_kmh"])
    ev["item_id"] = [RS.incident_key(a, b, c) for a, b, c in zip(ev["session_id"], ev["kind"], ev["start"])]
    rv = RS.review_map("incident")
    ev["status"] = [rv.get(i, {}).get("status", "Open") for i in ev["item_id"]]
    ev["reviewer"] = [rv.get(i, {}).get("reviewer", "") for i in ev["item_id"]]
    ev["note"] = [rv.get(i, {}).get("note", "") for i in ev["item_id"]]
    ev["date"] = ev["start"].dt.date
    return ev


def page_incidents(bundle: C.Bundle, weights, r0) -> None:
    st.title("Incidents & Review")
    st.caption("Incidents that need a decision. Open one, record what you found, and alert the supervisor if needed.")
    t_queue, t_trips = st.tabs(["Incident queue", "Unusual trips"])

    with t_queue:
        ev = _incident_frame(bundle)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1.2, 1.3, 1.1, 1.2])
            vehicles = sorted(ev["vehicle_id"].unique().tolist(), key=lambda x: (len(x), x))
            sel_v = c1.multiselect("Vehicles", vehicles, default=vehicles, key="inc_v")
            dmin, dmax = ev["date"].min(), ev["date"].max()
            rng = c2.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="inc_d")
            sel_p = c3.multiselect("Priority", A.PRIORITIES, default=["High", "Medium"], key="inc_p")
            statuses = ["Open"] + RS.STATUSES
            sel_s = c4.multiselect("Status", statuses, default=["Open", "Follow-up needed"], key="inc_s")
        lo, hi = (rng[0], rng[-1]) if isinstance(rng, (tuple, list)) and len(rng) else (dmin, dmax)
        base = ev[ev["vehicle_id"].isin(sel_v) & (ev["date"] >= lo) & (ev["date"] <= hi)]
        f = base[base["priority"].isin(sel_p) & base["status"].isin(sel_s)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Open, high priority", int(((base["status"] == "Open") & (base["priority"] == "High")).sum()))
        k2.metric("Open, all priorities", int((base["status"] == "Open").sum()))
        k3.metric("Confirmed", int((base["status"] == "Confirmed").sum()))
        k4.metric("False alarms", int((base["status"] == "False alarm").sum()))

        if f.empty:
            st.info("No incidents match these filters.")
        else:
            order = {p: i for i, p in enumerate(A.PRIORITIES)}
            f = f.assign(_o=f["priority"].map(order)).sort_values(["_o", "start"], ascending=[True, False])
            tbl = f[["priority", "status", "start", "vehicle_id", "event", "detail", "reviewer", "note"]].copy()
            tbl["start"] = tbl["start"].map(fmt_ts)
            tbl = tbl.rename(columns={"priority": "Priority", "status": "Status", "start": "Time", "vehicle_id": "Vehicle ID",
                                      "event": "Event", "detail": "Detail", "reviewer": "Reviewed by", "note": "Note"})
            show_df(tbl, height=300)
            st.download_button("Download queue (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                               file_name="incident_queue.csv", mime="text/csv", key="dl_queue")

            st.subheader("Review an incident")
            pick = f.head(300)
            labels = {r.item_id: f"{r.priority} | {fmt_ts(r.start)} | vehicle {r.vehicle_id} | {r.event} - {r.detail}"
                      for r in pick.itertuples()}
            iid = st.selectbox("Incident", list(labels), format_func=lambda i: labels[i], key="inc_pick")
            row = pick[pick["item_id"] == iid].iloc[0]
            left, right = st.columns([1.5, 1])
            with left:
                show_fig(route_figure(bundle, row["session_id"]), key="inc_route")
                st.caption(f"Trip {row['session_id']}: all incidents on this trip are marked on the map.")
            with right:
                st.markdown(f"**{row['event']}** - {row['detail']}")
                st.caption(f"Vehicle {row['vehicle_id']} | {fmt_ts(row['start'])} | priority {row['priority']} "
                           f"| status {row['status']}")
                _review_form(iid, "incident", dict(status=row["status"] if row["status"] != "Open" else "",
                                                   note=row["note"], reviewer=row["reviewer"]), key="inc")
                msg = A.alert_text(row["vehicle_id"], row["event"], row["detail"], fmt_ts(row["start"]), row["priority"])
                with st.expander("Alert the supervisor"):
                    st.code(msg, language=None)
                    if st.button("Send alert (Telegram)", key="send_alert"):
                        mode, status, note = A.dispatch(msg)
                        RS.log_alert(iid, "Telegram", mode, status, msg)
                        if mode == "LIVE_API" and status == "DELIVERED":
                            st.success("Alert delivered to the Telegram group.")
                        else:
                            st.warning(f"Alert logged but not delivered live ({mode}, {status}). {note}")

        with st.expander("Alert history"):
            log = RS.load_alert_log()
            if log.empty:
                st.caption("No alerts sent yet.")
            else:
                show_df(log.drop(columns=["message"]).rename(columns={"sent_at": "Sent (MYT)", "item_id": "Item",
                                                                       "channel": "Channel", "mode": "Mode",
                                                                       "status": "Status"}), height=220)

    with t_trips:
        st.caption("Trips whose overall driving pattern differs from most others (found by a machine-learning model, "
                   "not by the rules). Useful for catching things the rules do not describe.")
        try:
            ml = get_ml(bundle, tuple(sorted(bundle.params.items())))
        except Exception as exc:                                   # scikit-learn missing
            st.info(f"Unusual-trip detection is unavailable: {exc}. Run: pip install scikit-learn scipy")
            return
        st.caption(f"Flagged by risk model v{ml.meta.get('version', '?')} "
                   f"(trained {str(ml.meta.get('trained_at', ''))[:10]}). Manage versions in System & Models.")
        flagged = ml.table[ml.table["anomaly"] >= 90].copy()
        if flagged.empty:
            st.info("No unusual trips found.")
            return
        rv = RS.review_map("trip")
        flagged["status"] = [rv.get(RS.trip_key(i), {}).get("status", "Open") for i in flagged["session_id"]]
        flagged["reviewer"] = [rv.get(RS.trip_key(i), {}).get("reviewer", "") for i in flagged["session_id"]]
        flagged["note"] = [rv.get(RS.trip_key(i), {}).get("note", "") for i in flagged["session_id"]]
        flagged = flagged.sort_values("anomaly", ascending=False)
        tt = flagged[["session_id", "vehicle_id", "km", "reason", "score", "tier", "status", "reviewer", "note"]].rename(
            columns={"session_id": "Trip", "vehicle_id": "Vehicle ID", "reason": "What is unusual", "score": "DSS",
                     "tier": "Tier", "status": "Status", "reviewer": "Reviewed by", "note": "Note"}).round(1)
        show_df(tt, height=280)
        sid = st.selectbox("Trip", flagged["session_id"].tolist(), key="trip_pick",
                           format_func=lambda i: f"{i} - vehicle {flagged.set_index('session_id').loc[i, 'vehicle_id']}"
                                                 f" - {flagged.set_index('session_id').loc[i, 'reason']}")
        r = flagged.set_index("session_id").loc[sid]
        l2, r2 = st.columns([1.5, 1])
        with l2:
            show_fig(route_figure(bundle, sid), key="trip_route")
        with r2:
            st.markdown(f"**What is unusual:** {r['reason']}")
            st.caption(f"Rule-based score {r['score']:.0f} ({r['tier']}). Status: {r['status']}.")
            _review_form(RS.trip_key(sid), "trip", dict(status=r["status"] if r["status"] != "Open" else "",
                                                          note=r["note"], reviewer=r["reviewer"]), key="trip")


# ------------------------------------------------------------ driver performance
def page_performance(bundle: C.Bundle, weights, r0) -> None:
    st.title("Driver Performance")
    st.caption("How each driver is doing over time. Only days when the vehicle drove 50 km or more are counted.")
    scored = C.session_scores(bundle, weights, r0)
    allsess = P.driving(scored)
    if allsess.empty:
        st.info("No driving days available.")
        return
    dmin, dmax = allsess["date"].min(), allsess["date"].max()
    with st.container(border=True):
        c1, c2 = st.columns([1.2, 2])
        rng = c1.date_input("Period", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="perf_d")
        lo, hi = (rng[0], rng[-1]) if isinstance(rng, (tuple, list)) and len(rng) else (dmin, dmax)
        vehicles = sorted(allsess["vehicle_id"].unique().tolist(), key=lambda x: (len(x), x))
        sel_v = c2.multiselect("Drivers (vehicles)", vehicles, default=vehicles, key="perf_v")
    lim = bundle.params["speed_limit_kmh"]
    tbl = P.driver_table(scored, bundle.events, weights, r0, lo, hi, lim)
    tbl = tbl[tbl["vehicle_id"].isin(sel_v)]
    if tbl.empty:
        st.info("No driving days in this period.")
        return
    d_in = P.driving(scored, lo, hi)
    d_in = d_in[d_in["vehicle_id"].isin(sel_v)]
    attention = int(tbl["Tier"].isin(["Critical", "High Risk"]).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fleet mean DSS", f"{d_in['score'].mean():.0f}")
    k2.metric("Drivers needing attention", attention, help="Mean score in the Critical or High Risk tier.")
    k3.metric("Driving days", len(d_in))
    k4.metric("Distance", f"{d_in['km'].sum():,.0f} km")

    st.subheader("Score trend by week")
    trend = P.weekly_trend(scored, lo, hi)
    trend = trend[trend["vehicle_id"].isin(sel_v)]
    fleet = P.fleet_weekly(trend, d_in)
    enough = tbl[tbl["Data"] == ""]
    default_lines = (enough if len(enough) >= 3 else tbl).head(3)["vehicle_id"].tolist()
    lines = st.multiselect("Show these drivers on the chart", tbl["vehicle_id"].tolist(), default=default_lines, key="perf_lines")
    fig = go.Figure()
    for v in lines:
        g = trend[trend["vehicle_id"] == v]
        fig.add_trace(go.Scatter(x=g["week"], y=g["score"], mode="lines+markers", name=f"Vehicle {v}",
                                 line=dict(width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=fleet["week"], y=fleet["score"], mode="lines", name="Fleet average",
                             line=dict(width=4, color="#0f172a", dash="dot")))
    fig.update_layout(height=340, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Mean DSS", yaxis_range=[0, 100], legend=dict(orientation="h", y=1.12))
    show_fig(fig, key="perf_trend")

    st.subheader("Driver ranking")
    show = tbl.rename(columns={"vehicle_id": "Vehicle ID"}).copy()
    show["Trend (pts)"] = show["Trend (pts)"].round(1)
    show_df(show.round(1), height=min(420, 60 + 36 * len(show)), column_config={
        "Mean DSS": st.column_config.ProgressColumn("Mean DSS", min_value=0, max_value=100, format="%.0f")})
    st.caption("Trend = change in mean score between the driver's latest month and the month before it that has "
               "data (at least 2 driving days each). Drivers with fewer than 5 driving days are marked Limited "
               "data and should not be ranked firmly.")
    st.download_button("Download driver report (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="driver_performance.csv", mime="text/csv", key="dl_perf")

    st.subheader("What drives each driver's risk")
    mix = P.behaviour_mix(scored, bundle.events, weights, r0, lo, hi, lim)
    mix = mix[mix["vehicle_id"].isin(sel_v)]
    mix = mix.set_index("vehicle_id").loc[[v for v in tbl["vehicle_id"] if v in set(mix["vehicle_id"])]].reset_index() \
        if not mix.empty else mix
    fig2 = go.Figure()
    for k in R.DEFAULT_WEIGHTS:
        if (mix[k] > 0).any():
            fig2.add_trace(go.Bar(y="Vehicle " + mix["vehicle_id"].astype(str), x=mix[k], name=C.KIND_LABEL[k],
                                  orientation="h", marker_color=KIND_COLOR[k]))
    fig2.update_layout(barmode="stack", height=max(260, 34 * len(mix) + 90), template="plotly_white",
                       margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Risk points per 100 km",
                       legend=dict(orientation="h", y=1.12), yaxis=dict(autorange="reversed"))
    show_fig(fig2, key="perf_mix")

    st.subheader("Driver focus")
    v = st.selectbox("Driver (vehicle)", tbl["vehicle_id"].tolist(), key="perf_focus",
                     format_func=lambda x: f"Vehicle {x}")
    worst = d_in[d_in["vehicle_id"] == v].sort_values("score").head(8)
    wt = worst[["session_id", "start", "km", "events", "score", "tier"]].copy()
    wt["start"] = wt["start"].map(fmt_ts)
    wt = wt.rename(columns={"session_id": "Trip", "start": "Start", "km": "km", "events": "Incidents",
                            "score": "DSS", "tier": "Tier"}).round(1)
    st.markdown("**Lowest-scoring trips** (open them in History & Reports for the route and score breakdown)")
    show_df(wt, height=min(330, 60 + 36 * len(wt)))


# ------------------------------------------------------------- unusual-pattern flag
@st.cache_resource(show_spinner="Scoring trips with the risk model...")
def _score_view(_bundle, cache_key: tuple, version: int):
    scored = C.session_scores(_bundle, dict(R.DEFAULT_WEIGHTS), 10.0)
    mb = ML.load_model(version)
    return ML.ModelView(ML.score_with(mb, ML.session_features(_bundle.df), scored), mb.meta)


def _train_new_version(bundle: C.Bundle) -> dict:
    scored = C.session_scores(bundle, dict(R.DEFAULT_WEIGHTS), 10.0)
    sess = bundle.sessions
    rng = f"{sess['start'].min():%Y-%m-%d} to {sess['end'].max():%Y-%m-%d}"
    return ML.train_and_save(scored, ML.session_features(bundle.df), 10.0, date_range=rng)


def get_ml(bundle: C.Bundle, cache_key: tuple):
    """Trip table scored by the active saved model. Trains the first version automatically if none exists."""
    v = ML.active_version()
    if v is None:
        with st.spinner("Training the risk model for the first time (about 30 seconds)..."):
            v = _train_new_version(bundle)
    return _score_view(bundle, cache_key, int(v["version"]))


# --------------------------------------------------------------- system & models
def page_system(bundle: C.Bundle) -> None:
    st.title("System & Models")
    st.caption("Data coverage, the risk model that flags unusual trips, and the alert gateway.")
    df, sess = bundle.df, bundle.sessions
    real = sess[sess["pings"] >= C.MIN_PINGS_REPLAY]

    st.subheader("Data")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Telemetry records", f"{len(df):,}")
    k2.metric("Vehicles", df["vehicle_id"].nunique())
    k3.metric("Trips (vehicle-days)", len(real))
    k4.metric("Period", f"{sess['start'].min():%d %b} - {sess['end'].max():%d %b %Y}")
    gaps = df["dt"].dropna()
    gaps = gaps[gaps > 0]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Typical reporting interval", f"{gaps.median():.0f} s")
    q2.metric("Gaps longer than 10 min", f"{(gaps > 600).mean():.1%}")
    q3.metric("Implausible speeds (> 160 km/h)", f"{int((df['speed'] > 160).sum()):,}")
    q4.metric("Fuel jumps while moving (> 30 L)", f"{int(((df['dl'].abs() > 30) & (df['speed'] > 5)).sum()):,}",
              help="Large fuel-level changes while driving are treated as sensor noise, not theft.")
    cov = (real.groupby("vehicle_id").agg(Trips=("session_id", "size"), Records=("pings", "sum"), km=("km", "sum"),
                                         First=("start", "min"), Last=("end", "max"))
           .reset_index().rename(columns={"vehicle_id": "Vehicle ID"}))
    drv = real[real["km"] >= ML.MIN_ACTIVE_KM].groupby("vehicle_id").size().rename("Driving days")
    cov = cov.merge(drv, left_on="Vehicle ID", right_index=True, how="left").fillna({"Driving days": 0})
    cov["First"], cov["Last"] = cov["First"].dt.strftime("%Y-%m-%d"), cov["Last"].dt.strftime("%Y-%m-%d")
    cov["Driving days"] = cov["Driving days"].astype(int)
    st.markdown("**Coverage by vehicle**")
    show_df(cov.sort_values("Vehicle ID", key=lambda x: x.astype(int)).round(0), height=min(400, 60 + 36 * len(cov)))

    st.subheader("Risk model")
    st.caption("Learns normal driving patterns from the telemetry and flags unusual trips for review. "
               "Trained on the rule-based scores, because no crash or claims history is available.")
    versions = ML.list_versions()
    act = ML.active_version()
    if act is None:
        st.info("No model has been trained yet. It is trained automatically the first time Incidents & Review "
                "or History & Reports needs it, or you can train it now.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active version", f"v{act['version']}")
        m2.metric("Trained", str(act["trained_at"])[:16].replace("T", " "), help="Malaysia time")
        m3.metric("Learned from", f"{act['n_driving']} driving trips", f"{act['n_trips']} trips in total", delta_color="off")
        m4.metric("Vehicles", act["n_vehicles"])
        st.success(f"Model health: on vehicles it had never seen, it placed a trip within one risk tier of the "
                   f"rule-based result {act['rf_within1']:.0%} of the time (a model that always guesses the typical "
                   f"trip manages {act['base_within1']:.0%}). Trips it ranks as most unusual are shown in Incidents & Review.")
    c1, c2 = st.columns([1, 3])
    if c1.button("Train new version" if act else "Train model", key="train_btn"):
        with st.spinner("Training and validating (about 30 seconds)..."):
            try:
                meta = _train_new_version(bundle)
                st.success(f"Model v{meta['version']} trained and activated.")
            except ImportError:
                st.warning("Training needs scikit-learn and scipy. Run: pip install scikit-learn scipy")
            except ValueError as exc:
                st.warning(str(exc))
        st.rerun()
    if versions:
        vt = pd.DataFrame([{
            "Version": f"v{m['version']}" + (" (active)" if act and m["version"] == act["version"] else ""),
            "Trained (MYT)": str(m["trained_at"])[:16].replace("T", " "), "Data period": m.get("date_range", ""),
            "Driving trips": m["n_driving"], "Within one tier": f"{m['rf_within1']:.0%}",
            "Rank agreement": round(m["rf_spearman"], 2), "Avg score error": round(m["rf_mae"], 1)} for m in versions])
        show_df(vt, height=min(300, 60 + 36 * len(vt)))
        if len(versions) > 1:
            pick = st.selectbox("Switch active version", [m["version"] for m in versions],
                                index=[m["version"] for m in versions].index(act["version"]) if act else 0,
                                format_func=lambda v: f"v{v}", key="pick_version")
            if act and pick != act["version"] and st.button("Use this version", key="use_version"):
                ML.set_active(int(pick))
                st.rerun()

    st.subheader("Alert gateway")
    try:
        from src.services.telegram_service import TelegramNotificationService
        svc = TelegramNotificationService()
        live = svc.is_configured
    except Exception:
        live = False
    st.write("Telegram: " + ("connected" if live else "not configured - alerts are logged in simulation mode. "
                             "Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env to send them live."))
    if st.button("Send test message", key="test_alert"):
        mode, status, note = A.dispatch("Driver Risk Monitor: test message from System & Models.")
        (st.success if mode == "LIVE_API" and status == "DELIVERED" else st.warning)(
            f"Test result: {mode}, {status}. {note}")

    with st.expander("Detection settings in force"):
        show_df(pd.DataFrame({"Setting": list(bundle.params), "Value": list(bundle.params.values())}))


# ------------------------------------------------------------------- main
def main() -> None:
    page, weights, half_life, r0, params = sidebar()
    if page != "Live Monitor":
        st.session_state["play"] = False

    db = Path(DEFAULT_DB_PATH)
    try:
        bundle = get_bundle(str(db), db.stat().st_mtime if db.exists() else 0.0, tuple(sorted(params.items())))
    except C.DataUnavailable as exc:
        st.error(str(exc))
        st.stop()
        return
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        logger.exception("Failed to build telemetry bundle")
        st.error(f"Could not process telemetry: {exc}")
        st.stop()
        return

    try:
        if page == "Live Monitor":
            page_live(bundle, weights, half_life, r0)
        elif page == "Incidents & Review":
            page_incidents(bundle, weights, r0)
        elif page == "Driver Performance":
            page_performance(bundle, weights, r0)
        elif page == "System & Models":
            page_system(bundle)
        elif page == "Scoring Policy":
            page_lab(bundle, weights, half_life, r0)
        else:
            page_sessions(bundle, weights, half_life, r0)
    except C.DataUnavailable as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("RerunException", "StopException"):
            raise
        logger.exception("View failure")
        st.error(f"Something went wrong while rendering this page: {exc}")


main()
