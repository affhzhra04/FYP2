"""Explainable Risk Scoring View.

Interactive explainability, formulaic breakdown transparency,
mathematical audit steps, and risk weight simulation sandbox.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components.ui import (
    render_empty_state,
    render_metric_card,
    render_page_header,
)
from src.scoring import RiskScoringEngine
from src.services.data_service import TelematicsDataService


def render_explainable_scoring_page(service: TelematicsDataService) -> None:
    """Render Page: Explainable Risk Scoring."""
    # 1. Fetch Registered Vehicles
    vehicles = service.db.get_all_vehicles()
    if not vehicles:
        render_page_header(
            title="Explainable Driver Safety Scoring (DSS)",
            description="Deterministic scoring auditability and sensitivity simulation.",
            meta_badges=[("Registry", "Empty"), ("Status", "Awaiting Assets")],
        )
        render_empty_state(
            title="No Vehicles Registered",
            description="Register vehicles in Fleet Configuration to audit safety scores.",
        )
        return

    v_ids = [v["vehicle_id"] for v in vehicles]

    # Compact Vehicle Selector Strip
    sc1, sc2 = st.columns([1.2, 2.8])
    with sc1:
        selected_vid = st.selectbox(
            "Select Vehicle ID to Audit:",
            options=v_ids,
            index=0,
            key="es_target_vid",
            help="Choose vehicle to view its mathematical score deduction steps.",
        )

    # Page Header with selected asset
    render_page_header(
        title="Explainable Driver Safety Scoring (DSS)",
        description="Deterministic scoring auditability, transparent deduction formulas, and sensitivity analysis.",
        meta_badges=[
            ("Audited Asset", selected_vid),
            ("Scoring Range", "Bounded [0, 100]"),
            ("Audit Status", "Verified"),
        ],
    )

    # 2. Mathematical Formulation Display
    st.markdown("### Mathematical Formulation")
    st.markdown(
        r"""
$$\text{DSS} = \max\left(0.00, \;\min\left(100.00, \;100.00 - \sum_{c \in \mathcal{C}} \text{Penalty}_c\right)\right), \quad \text{where} \quad \text{Penalty}_c = \text{Count}_c \times \text{Base}_c \times \left(\frac{w_c}{3.0}\right) \times M_{\text{compound}}$$
        """
    )

    # 3. Interactive Weight Sandbox (What-If Analysis)
    st.markdown("### Sensitivity Analysis & Risk Weight Tuning Sandbox")

    default_weights = {
        "fuel_theft": 5.0,
        "moonlighting": 4.0,
        "speeding": 3.0,
        "harsh_driving": 2.0,
        "idling": 1.0,
    }

    with st.expander("Configure Dynamic Risk Category Weights (1.0 to 5.0)", expanded=False):
        w_cols = st.columns(5)
        custom_weights: Dict[str, float] = {}

        with w_cols[0]:
            w_fuel = st.slider("Fuel Theft (w1)", min_value=1.0, max_value=5.0, value=default_weights["fuel_theft"], step=0.5, key="es_w_fuel")
            custom_weights["fuel_theft"] = w_fuel
        with w_cols[1]:
            w_moon = st.slider("Moonlighting (w2)", min_value=1.0, max_value=5.0, value=default_weights["moonlighting"], step=0.5, key="es_w_moon")
            custom_weights["moonlighting"] = w_moon
        with w_cols[2]:
            w_speed = st.slider("Speeding (w3)", min_value=1.0, max_value=5.0, value=default_weights["speeding"], step=0.5, key="es_w_speed")
            custom_weights["speeding"] = w_speed
        with w_cols[3]:
            w_harsh = st.slider("Harsh Driving (w4)", min_value=1.0, max_value=5.0, value=default_weights["harsh_driving"], step=0.5, key="es_w_harsh")
            custom_weights["harsh_driving"] = w_harsh
        with w_cols[4]:
            w_idle = st.slider("Idling (w5)", min_value=1.0, max_value=5.0, value=default_weights["idling"], step=0.5, key="es_w_idle")
            custom_weights["idling"] = w_idle

        compound_mult = st.select_slider(
            "Interaction Compounding Multiplier (M):",
            options=[1.0, 1.25, 1.5, 1.75, 2.0],
            value=1.5,
            key="es_compound_mult",
            help="1.0 corresponds to standard linear baseline; 1.5 represents non-linear interaction compounding.",
        )

    # 4. Compute Audit Score Report
    score_report = service.calculate_vehicle_safety_score(
        vehicle_id=selected_vid,
        custom_weights=custom_weights,
        compound_multiplier=compound_mult,
    )

    # 5. Score Overview KPI Cards
    st.markdown(f"### Audit Breakdown for Vehicle '{selected_vid}'")

    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        render_metric_card(
            label="Driver Safety Score",
            value=f"{score_report.safety_score:.2f} / 100",
            subtitle="Bounded in [0.00, 100.00]",
        )
    with res_c2:
        tier_str = score_report.risk_tier.value if hasattr(score_report.risk_tier, "value") else str(score_report.risk_tier)
        render_metric_card(
            label="Risk Classification",
            value=tier_str.split(" ")[0],
            subtitle="Operating risk category",
        )
    with res_c3:
        render_metric_card(
            label="Total Penalty Points",
            value=f"-{score_report.total_deductions:.2f}",
            subtitle="Starting from baseline 100.0",
        )
    with res_c4:
        comp_str = f"Active ({compound_mult}x)" if score_report.compounding_applied else "None (1.0x)"
        render_metric_card(
            label="Compounding Status",
            value=comp_str,
            subtitle=f"{len(score_report.compounded_interactions)} Co-Occurrences",
        )

    # 6. Detailed Step-by-Step Penalty Deduction Audit Table
    st.markdown("### Itemized Penalty Breakdown & Arithmetic Verification")

    base_penalties = getattr(service.scoring_engine, "base_penalties", {})
    active_weights = custom_weights or getattr(service.scoring_engine, "weights", {})

    breakdown_data = []
    for cat, deduction in score_report.breakdown.items():
        cnt = score_report.infraction_counts.get(cat, 0)
        base_p = base_penalties.get(cat, 0.0)
        w = active_weights.get(cat, 3.0)
        norm_w = w / 3.0
        comp_factor = (
            compound_mult
            if score_report.compounding_applied and cnt > 0 and (
                (cat in ("speeding", "moonlighting") and any("MOONLIGHTING" in s for s in score_report.compounded_interactions))
                or (cat == "fuel_theft" and any("THEFT" in s for s in score_report.compounded_interactions))
                or (cat == "harsh_driving" and any("RECKLESS" in s for s in score_report.compounded_interactions))
            )
            else 1.0
        )

        breakdown_data.append({
            "Risk Category": cat.replace("_", " ").title(),
            "Raw Incidents": cnt,
            "Base (pts)": base_p,
            "Weight": w,
            "Norm Weight (w/3)": f"{norm_w:.3f}",
            "Multiplier": f"{comp_factor:.2f}x",
            "Calculated Penalty": f"-{deduction:.2f}",
        })

    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    tier_label = score_report.risk_tier.value if hasattr(score_report.risk_tier, "value") else str(score_report.risk_tier)
    st.info(
        f"**Arithmetic Summary:** "
        f"Score = 100.00 - ({score_report.total_deductions:.2f}) = **{score_report.safety_score:.2f}** "
        f"→ Categorized into **{tier_label}**."
    )

    # 7. Radar Chart of Supported Risk Dimensions
    st.markdown("### Multidimensional Risk Vector Profile")

    categories = ["Fuel Security", "Schedule Compliance", "Speed Control", "Kinematic Smoothness", "Idling Efficiency"]
    keys = ["fuel_theft", "moonlighting", "speeding", "harsh_driving", "idling"]

    dimension_scores = []
    for k in keys:
        deduction = score_report.breakdown.get(k, 0.0)
        dim_score = max(0.0, 100.0 - deduction)
        dimension_scores.append(dim_score)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=dimension_scores + [dimension_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.12)',
        line=dict(color='#2563eb', width=2),
        name=f"Vehicle {selected_vid}",
    ))

    benchmark = [75.0] * (len(categories) + 1)
    fig_radar.add_trace(go.Scatterpolar(
        r=benchmark,
        theta=categories + [categories[0]],
        mode='lines',
        line=dict(color='#dc2626', width=1.5, dash='dash'),
        name="Safety Benchmark (75.0)",
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#e2e8f0",
                linecolor="#e2e8f0",
            ),
            angularaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
            bgcolor="#ffffff",
        ),
        showlegend=True,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=320,
        margin=dict(l=30, r=30, t=25, b=25),
        font=dict(color="#334155", size=11),
    )

    st.plotly_chart(fig_radar, use_container_width=True)
