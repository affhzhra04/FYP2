"""Project evaluation view for the WIA3003 panel (not part of the company-facing product).

Run from the project root:   streamlit run src/evaluation.py

Shows how the scoring method and the machine-learning layer were evaluated: the DSS formula,
leave-one-vehicle-out validation, feature importance, anomaly detection and the limits of the
evidence. The product dashboard (src/dashboard.py) deliberately leaves all of this out.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.db_manager import DEFAULT_DB_PATH
from src import dash_core as C
from src import ml_layer as ML
from src import risk_engine as R

st.set_page_config(page_title="Project evaluation view", layout="wide")
st.markdown("<style>.block-container{padding-top:1.6rem;max-width:1500px}h1{font-size:1.7rem !important}</style>",
            unsafe_allow_html=True)


def show_fig(fig, key=None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


@st.cache_resource(show_spinner="Loading telemetry and running detectors...")
def get_bundle(db_path: str, mtime: float):
    return C.build_bundle(C.read_pings(db_path))


@st.cache_resource(show_spinner="Training models (leave-one-vehicle-out validation)...")
def get_ml(_bundle, cache_key: tuple):
    scored = C.session_scores(_bundle, dict(R.DEFAULT_WEIGHTS), 10.0)
    return ML.run_ml(scored, ML.session_features(_bundle.df), 10.0)


def page_method() -> None:
    st.title("Scoring method")
    st.caption("How the Driver Safety Score (DSS) is built from detected events.")
    st.latex(r"DSS = 100\cdot e^{-R/R_0},\qquad R=\frac{1}{D}\sum_{e}\frac{w_k}{3}\,b_k\,s_e")
    st.markdown(
        "- **w** is the user-tunable weight of the behaviour, **b** its base penalty, **s** the severity of the event "
        "(size and duration).\n"
        "- **D** is distance driven in units of 100 km (minimum 0.5), so session risk is per 100 km.\n"
        "- **R0** is the strictness. The live score applies a memory half-life instead of the 1/D normalisation.\n"
        "- Events: speeding, harsh driving, idling, route deviation, unauthorized stop, fuel loss, and fuel loss "
        "while off-route or at an unknown stop (compound event).")


def page_ml(bundle: C.Bundle) -> None:
    st.title("Project evaluation view")
    st.caption("A machine-learning layer next to the rules: it learns driving patterns from raw telemetry, "
               "and is tested on vehicles it has never seen.")
    try:
        res = get_ml(bundle, tuple(sorted(bundle.params.items())))
    except ImportError:
        st.warning("This page needs scikit-learn and scipy. Run: pip install scikit-learn scipy")
        return

    m = res.metrics
    act = m[m["Subset"].str.startswith("Active")].set_index("Model")
    rf, base = act.loc["Random Forest"], act.loc["Median baseline"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sessions evaluated", f"{res.n_active} driving / {res.n_total} total",
              help=f"Driving sessions cover at least {ML.MIN_ACTIVE_KM:.0f} km.")
    k2.metric("Tier within one step", f"{rf['Tier_within1']:.0%}", f"baseline {base['Tier_within1']:.0%}",
              delta_color="off")
    k3.metric("Rank agreement (Spearman)", f"{rf['Spearman']:.2f}", f"baseline {base['Spearman']:.2f}",
              delta_color="off")
    k4.metric("Average score error", f"{rf['MAE_DSS']:.1f} pts", f"baseline {base['MAE_DSS']:.1f}",
              delta_color="off")

    st.subheader("1. Can the model predict risk for a vehicle it has never seen?")
    st.caption("Leave-one-vehicle-out validation: train on nine vehicles, predict the tenth, repeat for every "
               "vehicle. The model sees only speed, acceleration, idle, night, turning, fuel and distance features, "
               "never the rule outputs. It predicts a continuous score instead of fixed thresholds.")
    mt = m.rename(columns={"MAE_DSS": "Avg score error (pts)", "Tier_exact": "Exact tier",
                           "Tier_within1": "Within one tier"})
    show_df(mt[["Model", "Subset", "Sessions", "Avg score error (pts)", "Spearman", "Exact tier",
                "Within one tier"]].round(3))

    a = res.table[res.table["km"] >= ML.MIN_ACTIVE_KM]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["score"], y=a["ml_score"], mode="markers", name="Session",
                             marker=dict(size=7, color="#2563eb", opacity=0.6),
                             hovertext="Session " + a["session_id"] + " (vehicle " + a["vehicle_id"] + ")"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect agreement",
                             line=dict(color="#94a3b8", dash="dash")))
    fig.update_layout(height=380, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Rule-based session DSS", yaxis_title="ML-predicted DSS (unseen vehicle)",
                      xaxis_range=[0, 100], yaxis_range=[0, 100], legend=dict(orientation="h", y=1.1))
    show_fig(fig, key="ml_scatter")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2. What drives the prediction")
        top = res.importance.head(10).sort_values("importance")
        fi = go.Figure(go.Bar(x=top["importance"], y=top["label"], orientation="h", marker_color="#2563eb"))
        fi.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="Feature importance (Random Forest)")
        show_fig(fi, key="ml_importance")
    with c2:
        st.subheader("Per vehicle (held out)")
        if res.per_vehicle.empty:
            st.info("Not enough driving sessions per vehicle.")
        else:
            pv = res.per_vehicle.rename(columns={"Active_sessions": "Driving sessions",
                                                 "MAE_DSS": "Avg score error (pts)"})
            show_df(pv.round(2), height=330)

    st.subheader("3. Anomaly detection (Isolation Forest, no labels)")
    ag = res.agreement
    if np.isnan(ag.get("spearman", np.nan)):
        st.info("Not enough driving sessions for anomaly detection.")
    else:
        st.caption(f"Agreement with the rules: Spearman {ag['spearman']:.2f}, "
                   f"{ag['top_overlap']:.0%} overlap between the 10% most anomalous and 10% riskiest sessions. "
                   f"{ag['n_ml_only']} sessions look anomalous to the model but are rated Excellent or Low Risk "
                   "by the rules and are worth a manual review.")
        flagged = res.table[res.table["ml_only_flag"]].sort_values("anomaly", ascending=False)
        if not flagged.empty:
            ft = flagged[["session_id", "vehicle_id", "km", "speed_p99", "idle_share", "night_share",
                          "anomaly", "score", "tier"]].rename(columns={
                "session_id": "Session", "vehicle_id": "Vehicle ID", "km": "km", "speed_p99": "Speed p99",
                "idle_share": "Idle share", "night_share": "Night share", "anomaly": "Anomaly percentile",
                "score": "Rule DSS", "tier": "Rule tier"})
            show_df(ft.round(2), height=min(300, 60 + 35 * len(ft)))

    with st.expander("Limits of this evaluation"):
        st.markdown(
            "- The prediction target is the rule-based score, because Innonics provided no crash or "
            "claims labels. The evaluation shows that behaviour patterns generalise across vehicles, "
            "not that the rules are correct.\n"
            "- Only 10 vehicles exist, so each held-out fold is one vehicle and the per-vehicle numbers "
            "are noisy.\n"
            "- Timestamps are device time as stored, so the night-driving feature depends on the clock.\n"
            "- The model is for ranking and review support. The rule-based score remains the audited, "
            "explainable one.")


def main() -> None:
    db = Path(DEFAULT_DB_PATH)
    try:
        bundle = get_bundle(str(db), db.stat().st_mtime if db.exists() else 0.0)
    except C.DataUnavailable as exc:
        st.error(str(exc))
        st.stop()
        return
    with st.sidebar:
        st.markdown("### Project evaluation view")
        page = st.radio("View", ["Machine-learning evaluation", "Scoring method"], label_visibility="collapsed")
    if page == "Scoring method":
        page_method()
    else:
        page_ml(bundle)


main()
