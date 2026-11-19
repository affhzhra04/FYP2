"""Fleet Risk Trends & Analytics View.

Interactive visualisations for risk tier distribution, category incident frequencies,
temporal anomaly trends, vehicle ranking, and baseline vs compounding model comparison.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components.ui import (
    render_empty_state,
    render_metric_card,
    render_page_header,
)
from src.services.data_service import TelematicsDataService


def render_risk_trends_page(service: TelematicsDataService) -> None:
    """Render Page: Risk Trends and Analytics."""
    # 1. Fetch Fleet Status & Anomaly Data
    fleet_df = service.get_fleet_status_table()
    anomalies_df = service.get_anomalies(limit=10000)

    if fleet_df.empty:
        render_page_header(
            title="Fleet Risk Trends & Analytics",
            description="Longitudinal safety score analytics, incident category distributions, and empirical evaluation.",
            meta_badges=[("Fleet", "Empty"), ("Status", "Awaiting Ingestion")],
        )
        render_empty_state(
            title="No Fleet Data Available",
            description="Register vehicles and ingest telemetry to visualize fleet risk trends.",
        )
        return

    mean_score = float(fleet_df["safety_score"].mean()) if "safety_score" in fleet_df.columns else 100.0
    crit_count = int(fleet_df["risk_tier"].astype(str).str.contains("Critical|High", case=False).sum()) if "risk_tier" in fleet_df.columns else 0
    total_anoms = len(anomalies_df)
    open_anoms = int((anomalies_df["status"] != "RESOLVED").sum()) if not anomalies_df.empty and "status" in anomalies_df.columns else 0

    render_page_header(
        title="Fleet Risk Trends & Analytics",
        description="Longitudinal safety score analytics, incident category distributions, and empirical evaluation.",
        meta_badges=[
            ("Monitored Assets", f"{len(fleet_df)} Units"),
            ("Logged Incidents", f"{total_anoms:,} Events"),
            ("Mean Safety Score", f"{mean_score:.1f}/100"),
        ],
    )

    # 2. Executive KPI Cards Strip
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            label="Fleet Mean Safety Score",
            value=f"{mean_score:.1f} / 100",
            subtitle="Across all registered assets",
        )
    with m2:
        render_metric_card(
            label="High / Critical Units",
            value=f"{crit_count} Assets",
            subtitle="Requires operational focus",
            delta=f"{crit_count} Flagged" if crit_count > 0 else "0 Flagged",
            delta_color="inverse" if crit_count > 0 else "normal",
        )
    with m3:
        render_metric_card(
            label="Total Logged Infractions",
            value=f"{total_anoms:,}",
            subtitle="Across operational history",
        )
    with m4:
        render_metric_card(
            label="Open Incident Queue",
            value=f"{open_anoms}",
            subtitle="Pending supervisor sign-off",
            delta=f"{open_anoms} Open" if open_anoms > 0 else "0 Open",
            delta_color="inverse" if open_anoms > 0 else "normal",
        )

    # 3. Risk Tier Distribution & Continuous Safety Score
    st.markdown("### Fleet Safety Score & Risk Tier Distribution")
    col_t1, col_t2 = st.columns([1, 1.2], gap="medium")

    with col_t1:
        tier_counts = fleet_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Vehicle Count"]

        color_map = {
            "Excellent (Compliant)": "#16a34a",
            "Low Risk": "#2563eb",
            "Moderate Risk": "#d97706",
            "High Risk": "#ea580c",
            "Critical (Immediate Audit Required)": "#dc2626",
        }

        fig_tier = px.pie(
            tier_counts,
            values="Vehicle Count",
            names="Risk Tier",
            hole=0.45,
            color="Risk Tier",
            color_discrete_map=color_map,
            title="Distribution by Documented Risk Tiers",
            template="plotly_white",
        )
        fig_tier.update_traces(textposition="inside", textinfo="percent+label")
        fig_tier.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            paper_bgcolor="#ffffff",
            font=dict(color="#334155", size=11),
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_t2:
        fig_hist = px.histogram(
            fleet_df,
            x="safety_score",
            nbins=15,
            title="Driver Safety Score (DSS) Distribution",
            labels={"safety_score": "Continuous Safety Score (0–100)"},
            color_discrete_sequence=["#0f172a"],
            template="plotly_white",
        )
        fig_hist.add_vline(x=90.0, line_dash="dot", line_color="#16a34a", annotation_text="Exc (90)")
        fig_hist.add_vline(x=75.0, line_dash="dot", line_color="#2563eb", annotation_text="Low (75)")
        fig_hist.add_vline(x=50.0, line_dash="dot", line_color="#d97706", annotation_text="Mod (50)")
        fig_hist.add_vline(x=25.0, line_dash="dot", line_color="#dc2626", annotation_text="Crit (25)")

        fig_hist.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155", size=11),
        )
        fig_hist.update_xaxes(gridcolor="#f1f5f9")
        fig_hist.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_hist, use_container_width=True)

    # 4. Incident Frequency by Category & Resolution Breakdown
    st.markdown("### Incident Category & Resolution Status")

    if anomalies_df.empty:
        st.info("No anomaly incident records logged in the database.")
    else:
        col_c1, col_c2 = st.columns([1.2, 1], gap="medium")
        with col_c1:
            cat_counts = anomalies_df["incident_type"].value_counts().reset_index()
            cat_counts.columns = ["Incident Category", "Total Count"]
            cat_counts["Incident Category"] = cat_counts["Incident Category"].str.replace("_", " ").str.title()

            fig_bar = px.bar(
                cat_counts,
                x="Total Count",
                y="Incident Category",
                orientation="h",
                title="Incident Counts by Category",
                color_discrete_sequence=["#1e293b"],
                template="plotly_white",
            )
            fig_bar.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#334155", size=11),
            )
            fig_bar.update_xaxes(gridcolor="#f1f5f9")
            fig_bar.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c2:
            status_counts = anomalies_df["status"].value_counts().reset_index()
            status_counts.columns = ["Resolution Status", "Count"]

            fig_stat = px.pie(
                status_counts,
                values="Count",
                names="Resolution Status",
                hole=0.45,
                title="Resolution Status Distribution",
                color="Resolution Status",
                color_discrete_map={"OPEN": "#dc2626", "RESOLVED": "#16a34a"},
                template="plotly_white",
            )
            fig_stat.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#ffffff",
                font=dict(color="#334155", size=11),
            )
            st.plotly_chart(fig_stat, use_container_width=True)

    # 5. Vehicle Fleet Ranking & Comparative Scoring Evaluation
    st.markdown("### Comparative Scoring Evaluation: Baseline vs. Compounding Interaction Model")
    st.caption(
        "Empirical evaluation: Baseline model uses an independent linear penalty formula, "
        "whereas the compounding model applies an interaction multiplier (1.5x) for co-occurring risk events."
    )

    v_ids = sorted([str(v) for v in fleet_df["vehicle_id"].unique()])
    c_sel, _ = st.columns([1.2, 2.8])
    with c_sel:
        comp_vid = st.selectbox("Select Vehicle for Model Comparison:", options=v_ids, index=0, key="rt_comp_vid")

    comp_res = service.compare_fyp1_vs_fyp2_scoring(vehicle_id=comp_vid)

    c_box1, c_box2, c_box3 = st.columns(3)
    with c_box1:
        render_metric_card(
            label="Baseline Linear Model",
            value=f"{comp_res['fyp1_score']:.1f} / 100",
            subtitle=f"Tier: {comp_res['fyp1_tier']} | Deductions: {comp_res['fyp1_deductions']:.1f} pts",
        )
    with c_box2:
        render_metric_card(
            label="Compounding Interaction Model",
            value=f"{comp_res['fyp2_score']:.1f} / 100",
            subtitle=f"Tier: {comp_res['fyp2_tier']} | Deductions: {comp_res['fyp2_deductions']:.1f} pts",
            delta=f"-{comp_res['delta_points']:.1f} pts" if comp_res['delta_points'] > 0 else "0.0 pts",
            delta_color="inverse" if comp_res['delta_points'] > 0 else "normal",
        )
    with c_box3:
        applied_str = "Active" if comp_res["compounding_applied"] else "Isolated"
        render_metric_card(
            label="Interaction Multiplier",
            value="1.50x",
            subtitle=f"Status: {applied_str} ({len(comp_res['compounding_interactions'])} pairs)",
        )

    # 6. Fleet Asset Ranking
    with st.expander("Fleet Assets Ranked by Safety Score", expanded=False):
        sorted_fleet = fleet_df.sort_values(by="safety_score", ascending=True)
        fig_rank = px.bar(
            sorted_fleet,
            x="safety_score",
            y="vehicle_id",
            orientation="h",
            title="Fleet Assets Ranked by Safety Score (Ascending)",
            labels={"safety_score": "Safety Score", "vehicle_id": "Vehicle ID"},
            color="safety_score",
            color_continuous_scale="RdYlGn",
            template="plotly_white",
        )
        fig_rank.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155", size=11),
        )
        fig_rank.update_xaxes(gridcolor="#f1f5f9")
        fig_rank.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_rank, use_container_width=True)
