"""Telemetry Signal Inspector & Kinematics View.

Enterprise time-series sensor analysis, kinematic differentials (ΔT, ΔV, ΔL),
data quality verification, and incident detection overlays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.components.ui import (
    render_empty_state,
    render_metric_card,
    render_page_header,
)
from src.services.data_service import TelematicsDataService


def render_signal_inspector_page(service: TelematicsDataService) -> None:
    """Render Page: Signal Inspector."""
    # 1. Fetch Registered Vehicles
    vehicles = service.db.get_all_vehicles()
    if not vehicles:
        render_page_header(
            title="Telemetry Signal Inspector & Kinematics",
            description="Fine-grained time-series sensor analysis and kinematic diagnostics.",
            meta_badges=[("Registry", "Empty"), ("Status", "Awaiting Assets")],
        )
        render_empty_state(
            title="No Vehicles Registered",
            description="Register vehicles in Fleet Configuration to inspect telemetry signals.",
        )
        return

    v_ids = [v["vehicle_id"] for v in vehicles]

    # Compact Control Strip
    c1, c2, c3 = st.columns([1.4, 1.2, 1.0])
    with c1:
        selected_vid = st.selectbox(
            "Select Vehicle ID to Inspect:",
            options=v_ids,
            index=0,
            key="si_target_vid",
            help="Choose vehicle to query stored time-series pings.",
        )
    with c2:
        max_pings = st.select_slider(
            "Max Telemetry Points:",
            options=[500, 1000, 2500, 5000, 10000],
            value=2500,
            key="si_max_points",
            help="Limits returned frames for optimal rendering performance.",
        )
    with c3:
        st.write("")
        st.write("")
        include_anomalies_overlay = st.checkbox(
            "Overlay Incidents",
            value=True,
            key="si_overlay_chk",
            help="Mark detected anomalies as incident markers on the timeline.",
        )

    # 2. Query Time-Series Telemetry
    telemetry_df = service.get_vehicle_telemetry(vehicle_id=selected_vid, limit=max_pings)

    if telemetry_df.empty:
        render_page_header(
            title="Telemetry Signal Inspector & Kinematics",
            description="Fine-grained time-series sensor analysis and kinematic diagnostics.",
            meta_badges=[("Target Asset", selected_vid), ("Status", "No Telemetry")],
        )
        render_empty_state(
            title=f"No Telemetry Pings for Vehicle '{selected_vid}'",
            description=f"Vehicle '{selected_vid}' exists in registry but has zero recorded telemetry frames.",
        )
        return

    total_records = len(telemetry_df)
    min_date = telemetry_df["timestamp"].min().strftime("%Y-%m-%d")
    max_date = telemetry_df["timestamp"].max().strftime("%Y-%m-%d")

    # Render Header with Accurate Metadata
    render_page_header(
        title="Telemetry Signal Inspector & Kinematics",
        description="Fine-grained time-series sensor analysis and kinematic diagnostics.",
        meta_badges=[
            ("Target Asset", selected_vid),
            ("Frames Loaded", f"{total_records:,}"),
            ("Date Range", f"{min_date} to {max_date}"),
        ],
    )

    # 3. Summary Signal Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        avg_speed = telemetry_df["speed"].mean()
        max_speed = telemetry_df["speed"].max()
        render_metric_card(
            label="Avg / Max Velocity",
            value=f"{avg_speed:.1f} / {max_speed:.1f} km/h",
            subtitle="Raw physical speed sensor",
        )
    with m2:
        fuel_start = telemetry_df["fuel_level"].iloc[0] if pd.notna(telemetry_df["fuel_level"].iloc[0]) else 0.0
        fuel_end = telemetry_df["fuel_level"].iloc[-1] if pd.notna(telemetry_df["fuel_level"].iloc[-1]) else 0.0
        delta_net_fuel = fuel_end - fuel_start
        render_metric_card(
            label="Fuel Level Range",
            value=f"{fuel_start:.1f}L → {fuel_end:.1f}L",
            subtitle=f"Net delta: {delta_net_fuel:+.1f} Liters",
        )
    with m3:
        if "ignition" in telemetry_df.columns:
            on_ratio = (telemetry_df["ignition"] == 1).mean() * 100.0
            render_metric_card(
                label="Engine Duty Cycle",
                value=f"{on_ratio:.1f}% ON",
                subtitle="Operational ignition ratio",
            )
        else:
            render_metric_card(label="Ignition Status", value="Not Recorded")
    with m4:
        time_span = telemetry_df["timestamp"].max() - telemetry_df["timestamp"].min()
        hours_span = time_span.total_seconds() / 3600.0 if time_span.total_seconds() > 0 else 1.0
        render_metric_card(
            label="Operational Span",
            value=f"{hours_span:.1f} Hours",
            subtitle="Session window duration",
        )

    # 4. Interactive Time-Series Plotly Chart: Speed & Fuel Signals
    st.markdown("### Sensor Signals: Kinematic Speed & Ultrasonic Fuel Trajectories")

    # Fetch vehicle anomalies for overlay
    anomalies_df = pd.DataFrame()
    if include_anomalies_overlay:
        anomalies_df = service.get_anomalies(vehicle_id=selected_vid, limit=500)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Raw Speed Trajectory (km/h)", "Ultrasonic Fuel Level Sensor (Liters)"),
    )

    # Subplot 1: Speed
    fig.add_trace(
        go.Scatter(
            x=telemetry_df["timestamp"],
            y=telemetry_df["speed"],
            name="Speed (km/h)",
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            hovertemplate="Time: %{x}<br>Speed: %{y:.1f} km/h<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Reference Line (80 km/h)
    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Speed Threshold (80 km/h)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Subplot 2: Fuel Level
    fig.add_trace(
        go.Scatter(
            x=telemetry_df["timestamp"],
            y=telemetry_df["fuel_level"],
            name="Fuel Level (L)",
            mode="lines",
            line=dict(color="#059669", width=1.7),
            hovertemplate="Time: %{x}<br>Fuel: %{y:.2f} L<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Incident Overlays
    if include_anomalies_overlay and not anomalies_df.empty:
        anom_ts = pd.to_datetime(anomalies_df["timestamp"])
        min_t, max_t = telemetry_df["timestamp"].min(), telemetry_df["timestamp"].max()
        valid_anom = anomalies_df[(anom_ts >= min_t) & (anom_ts <= max_t)]

        if not valid_anom.empty:
            speed_anoms = valid_anom[valid_anom["incident_type"].str.contains("SPEED", na=False)]
            fuel_anoms = valid_anom[valid_anom["incident_type"].str.contains("FUEL", na=False)]

            if not speed_anoms.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(speed_anoms["timestamp"]),
                        y=speed_anoms["magnitude"],
                        mode="markers",
                        name="Speeding Incident",
                        marker=dict(symbol="triangle-up", size=9, color="#dc2626"),
                        hovertemplate="Incident: Speeding<br>Time: %{x}<br>Magnitude: %{y:.1f} km/h<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            if not fuel_anoms.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(fuel_anoms["timestamp"]),
                        y=[telemetry_df["fuel_level"].min()] * len(fuel_anoms),
                        mode="markers",
                        name="Fuel Theft Incident",
                        marker=dict(symbol="x", size=10, color="#d97706", line=dict(width=2)),
                        hovertemplate="Incident: Fuel Drop<br>Time: %{x}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

    fig.update_layout(
        height=420,
        margin=dict(l=35, r=15, t=25, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Computed Kinematic Differentials (ΔT, ΔV, ΔL)
    st.markdown("### Computed Kinematic Differentials (ΔT, ΔV, ΔL)")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        fig_dt = px.histogram(
            telemetry_df[telemetry_df["delta_time"] < 120],
            x="delta_time",
            nbins=30,
            title="Sampling Interval Distribution (ΔT Seconds)",
            labels={"delta_time": "ΔT (Seconds)"},
            color_discrete_sequence=["#0f172a"],
            template="plotly_white",
        )
        fig_dt.update_layout(
            margin=dict(l=20, r=20, t=32, b=20),
            height=220,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155", size=11),
        )
        fig_dt.update_xaxes(gridcolor="#f1f5f9")
        fig_dt.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_dt, use_container_width=True)

    with col_k2:
        fig_dv = px.histogram(
            telemetry_df[(telemetry_df["delta_speed"] > -30) & (telemetry_df["delta_speed"] < 30)],
            x="delta_speed",
            nbins=40,
            title="Speed Velocity Delta Distribution (ΔV km/h)",
            labels={"delta_speed": "ΔV (km/h)"},
            color_discrete_sequence=["#2563eb"],
            template="plotly_white",
        )
        fig_dv.update_layout(
            margin=dict(l=20, r=20, t=32, b=20),
            height=220,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155", size=11),
        )
        fig_dv.update_xaxes(gridcolor="#f1f5f9")
        fig_dv.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_dv, use_container_width=True)

    # 6. Data Quality & Tabular Telemetry
    with st.expander("Sensor Sanity Audit & Tabular Telemetry Records", expanded=False):
        missing_coords = int(telemetry_df["latitude"].isna().sum() + telemetry_df["longitude"].isna().sum())
        zero_delta_time = int((telemetry_df["delta_time"] <= 0).sum()) if "delta_time" in telemetry_df.columns else 0
        duplicate_ts = int(telemetry_df["timestamp"].duplicated().sum())

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Total Frames", f"{total_records:,}")
        with d2:
            st.metric("Missing Coords", missing_coords)
        with d3:
            st.metric("Interval Errors (ΔT ≤ 0)", zero_delta_time)
        with d4:
            st.metric("Duplicate Timestamps", duplicate_ts)

        st.dataframe(
            telemetry_df[["timestamp", "speed", "fuel_level", "ignition", "latitude", "longitude", "delta_time", "delta_speed", "delta_fuel"]],
            use_container_width=True,
            hide_index=True,
            height=240,
        )
