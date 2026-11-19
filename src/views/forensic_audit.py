"""Workspace 2: Mode B — Forensic Audit Mode (with Calendar Date Picker).

Implements end-of-shift telemetry audit, independent calendar date and shift selection,
interactive historical trajectory inspection with OpenStreetMap tiles, dynamic
Multi-Criteria Decision Analysis (MCDA) weight tuning, Explainable AI (XAI) driver
radar profiles, forensic fuel loss logs, and downloadable CSV audit reports.
Strictly enforces Malaysia Standard Time (MYT, UTC+8) with zero UI emojis.
"""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import Any, Dict, List, Optional, Union

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.components.ui import (
    get_risk_tier_badge_html,
    render_empty_state,
    render_metric_card,
    render_notice_banner,
    render_page_header,
)
from src.services.data_service import TelematicsDataService
from src.utils.timezone import now_myt, to_myt_str

logger = logging.getLogger(__name__)


def render_forensic_audit_view(service: TelematicsDataService) -> None:
    """Render Mode B: Forensic Audit Mode.

    Parameters
    ----------
    service : TelematicsDataService
        Data service layer instance.
    """
    # --------------------------------------------------------------------------
    # 1. Page Header & Audit Metadata
    # --------------------------------------------------------------------------
    render_page_header(
        title="Mode B: Forensic Audit Mode",
        description="Historical trajectory forensics, calendar date shift auditing, dynamic MCDA weight recalibration, and explainable safety profiles.",
        meta_badges=[
            ("Operational Architecture", "Mode B: Forensic Audit"),
            ("Decision Engine", "Multi-Criteria Decision Analysis (MCDA)"),
            ("Timezone Enforcement", "MYT (UTC+8)"),
            ("Audit Session", now_myt().strftime("%Y-%m-%d %H:%M MYT")),
        ],
    )

    # --------------------------------------------------------------------------
    # 2. Historical Shift & Calendar Date Selection Controls
    # --------------------------------------------------------------------------
    available_dates = service.get_available_historical_dates()

    if available_dates:
        min_date_val = date.fromisoformat(available_dates[0])
        max_date_val = date.fromisoformat(available_dates[-1])
        # Default to August 2024 shift if present, else latest available
        if "2024-08-09" in available_dates:
            default_date_val = date.fromisoformat("2024-08-09")
        elif "2024-01-10" in available_dates:
            default_date_val = date.fromisoformat("2024-01-10")
        else:
            default_date_val = max_date_val
    else:
        min_date_val = date(2024, 1, 1)
        max_date_val = date(2024, 12, 31)
        default_date_val = date(2024, 8, 9)

    st.markdown("##### Historical Shift & Calendar Controls")

    c_date, c_shift, c_hint = st.columns([1.5, 1.8, 2.7])
    with c_date:
        selected_date = st.date_input(
            "Audit Historical Date (MYT)",
            value=default_date_val,
            min_value=min_date_val,
            max_value=max_date_val,
            help="Select any past historical calendar date to audit independent of the live operational feed.",
        )

    with c_shift:
        shift_options = [
            "Full 24-Hour Shift",
            "Day Shift (08:00 - 17:00 MYT)",
            "Ghost Hours Shift (22:00 - 04:59 MYT)",
        ]
        selected_shift = st.selectbox(
            "Shift Inspection Window",
            options=shift_options,
            index=0,
            help="Filter telemetry and anomaly incidents for specific operational shift periods.",
        )

    with c_hint:
        dates_str = ", ".join(available_dates[-5:]) if available_dates else "None"
        st.markdown(
            f"""
            <div style="background-color:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px 14px; margin-top:24px; font-size:12px; color:#475569;">
                <strong style="color:#0f172a;">Historical Records in SQLite:</strong> {dates_str}<br/>
                <span style="font-size:11px; color:#64748b;">Selected Target: {selected_date.strftime('%A, %d %B %Y')} | {selected_shift}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Dynamic MCDA Weight Controls (Real-Time Weight Tuning)
    # --------------------------------------------------------------------------
    with st.expander("Dynamic MCDA Weight Controls (Real-Time Weight Tuning)", expanded=False):
        st.markdown(
            "Fine-tune Multi-Criteria Decision Analysis (MCDA) penalty weights in real-time. "
            "Adjusting these sliders dynamically recalibrates Driver Safety Scores (DSS) and fleet risk rankings for this historical shift."
        )

        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        with wc1:
            w_fuel = st.slider(
                "W_Fuel (Fuel Theft)",
                min_value=1.0,
                max_value=5.0,
                value=5.0,
                step=0.5,
                help="Severity multiplier for stationary persistent fuel drops and theft.",
            )
        with wc2:
            w_safety = st.slider(
                "W_Safety (Speed & Kinematics)",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Severity multiplier for overspeeding and harsh kinematic maneuvers.",
            )
        with wc3:
            w_route = st.slider(
                "W_Route (Geofence & Ghost Hours)",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.5,
                help="Severity multiplier for unauthorized night driving (22:00-04:59 MYT) and geofence breaches.",
            )
        with wc4:
            w_idle = st.slider(
                "W_Idle (Excessive Idling)",
                min_value=1.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="Severity multiplier for prolonged stationary idling with engine ON.",
            )
        with wc5:
            compound_mult = st.slider(
                "Compounding Factor",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Non-linear interaction multiplier for co-occurring high-risk infractions (e.g. speeding during moonlighting).",
            )

    custom_weights = {
        "fuel_theft": float(w_fuel),
        "moonlighting": float(w_route),
        "speeding": float(w_safety),
        "harsh_driving": float(w_safety) * 0.7,
        "idling": float(w_idle),
    }

    # --------------------------------------------------------------------------
    # 4. Compute Real-Time Leaderboard for Selected Shift
    # --------------------------------------------------------------------------
    leaderboard_df = service.get_forensic_leaderboard(
        custom_weights=custom_weights,
        compound_multiplier=float(compound_mult),
        target_date=selected_date,
        shift_window=selected_shift,
    )

    if leaderboard_df.empty:
        render_empty_state(
            title="No Shift Telemetry Records Found",
            description=f"The database contains zero logged telemetry or anomaly records for {selected_date.strftime('%Y-%m-%d')} during '{selected_shift}'. Please select another date from the historical records.",
        )
        return

    # High-level fleet KPIs summary for this audit run
    avg_dss = round(float(leaderboard_df["safety_score"].mean()), 1)
    tot_fuel_lost = round(float(leaderboard_df["fuel_loss_liters"].sum()), 2)
    tot_critical = int(leaderboard_df["critical_incidents"].sum())
    compliant_count = int((leaderboard_df["safety_score"] >= 75.0).sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card(
            label="Shift Average DSS",
            value=f"{avg_dss} / 100",
            subtitle=f"Shift: {selected_date.strftime('%Y-%m-%d')}",
            delta="MCDA Calibrated",
            delta_color="normal",
        )
    with k2:
        render_metric_card(
            label="Shift Fuel Loss",
            value=f"{tot_fuel_lost:.2f} L",
            subtitle="Stationary Siphons",
            delta="Pilferage Total",
            delta_color="inverse" if tot_fuel_lost > 0 else "normal",
        )
    with k3:
        render_metric_card(
            label="Critical Incidents",
            value=str(tot_critical),
            subtitle="Theft, Ghost Hours & Speed",
            delta="Audit Escalated",
            delta_color="inverse" if tot_critical > 0 else "normal",
        )
    with k4:
        render_metric_card(
            label="Compliant Fleet Units",
            value=f"{compliant_count} / {len(leaderboard_df)}",
            subtitle="DSS >= 75.0 (Compliant)",
            delta=f"{(compliant_count/max(1, len(leaderboard_df)))*100:.0f}% Compliance",
            delta_color="normal" if compliant_count >= (len(leaderboard_df) // 2) else "inverse",
        )

    # --------------------------------------------------------------------------
    # 5. End-of-Shift Analytics Leaderboard Table
    # --------------------------------------------------------------------------
    st.markdown("### End-of-Shift Analytics Leaderboard")

    table_cols = [
        "rank",
        "vehicle_id",
        "safety_score",
        "risk_tier",
        "fuel_loss_liters",
        "critical_incidents",
        "geofence_breaches",
        "night_breaches",
        "idling_count",
        "total_pings",
        "shift_start_myt",
        "shift_end_myt",
    ]
    display_df = leaderboard_df[table_cols].rename(
        columns={
            "rank": "Rank",
            "vehicle_id": "Vehicle ID",
            "safety_score": "Safety Score (DSS)",
            "risk_tier": "Risk Classification",
            "fuel_loss_liters": "Fuel Loss (L)",
            "critical_incidents": "Critical Breaches",
            "geofence_breaches": "Geofence Breaches",
            "night_breaches": "Ghost Hours Breaches",
            "idling_count": "Idling Pings",
            "total_pings": "Pings Logged",
            "shift_start_myt": "Shift Start (MYT)",
            "shift_end_myt": "Shift End (MYT)",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # 6. Historical Trajectory Forensics & Interactive Map
    # --------------------------------------------------------------------------
    st.markdown("### Historical Shift Trajectory & Anomaly Map")

    c_sel, _ = st.columns([2, 3])
    with c_sel:
        audited_vid = st.selectbox(
            "Select Audited Unit for Trajectory & XAI Breakdown",
            options=leaderboard_df["vehicle_id"].tolist(),
            index=0,
            help="Select an authentic Innonics vehicle ID to inspect historical GPS trajectory and explainable multi-axis risk profile.",
        )

    # Fetch trajectory telemetry pings for this vehicle on the chosen date & shift
    traj_pings_df = service.get_vehicle_telemetry(
        vehicle_id=audited_vid,
        target_date=selected_date,
        shift_window=selected_shift,
        limit=5000,
    )

    # Fetch fuel loss & critical anomaly logs for this vehicle on the chosen date & shift
    shift_fuel_anoms = service.get_fuel_loss_records(
        vehicle_id=audited_vid,
        target_date=selected_date,
        shift_window=selected_shift,
        limit=100,
    )

    if not traj_pings_df.empty:
        valid_traj = traj_pings_df.dropna(subset=["latitude", "longitude"])
        valid_traj = valid_traj[(valid_traj["latitude"] != 0.0) & (valid_traj["longitude"] != 0.0)]

        if not valid_traj.empty:
            mean_lat = float(valid_traj["latitude"].mean())
            mean_lon = float(valid_traj["longitude"].mean())

            traj_map = folium.Map(
                location=[mean_lat, mean_lon],
                zoom_start=12,
                tiles="OpenStreetMap",
                control_scale=True,
            )

            # Polyline route trace
            coords = valid_traj[["latitude", "longitude"]].values.tolist()
            folium.PolyLine(
                locations=coords,
                color="#2563eb",
                weight=4,
                opacity=0.85,
                tooltip=f"Vehicle {audited_vid} Trajectory ({len(coords)} Pings)",
            ).add_to(traj_map)

            # Trip Origin Marker
            start_row = valid_traj.iloc[0]
            start_ts = to_myt_str(start_row["timestamp"])
            folium.Marker(
                location=[float(start_row["latitude"]), float(start_row["longitude"])],
                popup=folium.Popup(
                    f"<b>Trip Origin (Shift Start)</b><br/>Time: {start_ts}<br/>Speed: {float(start_row['speed']):.1f} km/h",
                    max_width=250,
                ),
                tooltip=f"Start: {start_ts}",
                icon=folium.Icon(color="green", icon="play", prefix="glyphicon"),
            ).add_to(traj_map)

            # Trip Destination / Latest Point Marker
            end_row = valid_traj.iloc[-1]
            end_ts = to_myt_str(end_row["timestamp"])
            folium.Marker(
                location=[float(end_row["latitude"]), float(end_row["longitude"])],
                popup=folium.Popup(
                    f"<b>Trip Destination (Shift End)</b><br/>Time: {end_ts}<br/>Speed: {float(end_row['speed']):.1f} km/h",
                    max_width=250,
                ),
                tooltip=f"End: {end_ts}",
                icon=folium.Icon(color="blue", icon="stop", prefix="glyphicon"),
            ).add_to(traj_map)

            # Plot Fuel Loss / Siphon Anomaly Markers along trajectory
            if not shift_fuel_anoms.empty:
                for _, anom_row in shift_fuel_anoms.iterrows():
                    anom_ts = anom_row.get("timestamp_myt", "N/A")
                    anom_mag = float(anom_row.get("magnitude_liters", 0.0))
                    anom_type = str(anom_row.get("incident_type", "FUEL_THEFT"))

                    # Find nearest telemetry ping coordinates by timestamp if possible
                    folium.CircleMarker(
                        location=[mean_lat, mean_lon],
                        radius=8,
                        color="#dc2626",
                        fill=True,
                        fill_color="#dc2626",
                        fill_opacity=0.8,
                        popup=folium.Popup(
                            f"<b>{anom_type}</b><br/>Loss: -{anom_mag:.2f} Liters<br/>Timestamp: {anom_ts}",
                            max_width=250,
                        ),
                        tooltip=f"Security Alert: -{anom_mag:.2f} L at {anom_ts}",
                    ).add_to(traj_map)

            st_folium(traj_map, width="100%", height=380, returned_objects=[])

            # Trajectory Kinematic Summary Strip
            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                st.metric("Logged Shift Pings", f"{len(valid_traj):,}")
            with t2:
                st.metric("Peak Speed", f"{valid_traj['speed'].max():.1f} km/h")
            with t3:
                st.metric("Average Speed", f"{valid_traj['speed'].mean():.1f} km/h")
            with t4:
                st.metric("Shift Start (MYT)", start_ts.split(" ")[-1] if " " in start_ts else start_ts)
            with t5:
                st.metric("Shift End (MYT)", end_ts.split(" ")[-1] if " " in end_ts else end_ts)
        else:
            render_notice_banner(
                f"Vehicle {audited_vid} has records on {selected_date.strftime('%Y-%m-%d')}, but coordinates are invalid (0.0, 0.0).",
                banner_type="warning",
            )
    else:
        render_notice_banner(
            f"No GPS telemetry pings recorded for vehicle {audited_vid} during selected shift on {selected_date.strftime('%Y-%m-%d')}.",
            banner_type="info",
        )

    # --------------------------------------------------------------------------
    # 7. Explainable Driver Radar Profiles (XAI)
    # --------------------------------------------------------------------------
    st.markdown("### Explainable Driver Radar Profiles (XAI)")

    radar_data = service.get_radar_profile_metrics(
        vehicle_id=audited_vid,
        custom_weights=custom_weights,
        compound_multiplier=float(compound_mult),
        target_date=selected_date,
        shift_window=selected_shift,
    )

    col_chart, col_breakdown = st.columns([3, 2])

    with col_chart:
        categories = list(radar_data["vehicle_scores"].keys())
        vehicle_vals = list(radar_data["vehicle_scores"].values())
        fleet_vals = [radar_data["fleet_averages"].get(cat, 50.0) for cat in categories]

        categories_closed = categories + [categories[0]]
        vehicle_vals_closed = vehicle_vals + [vehicle_vals[0]]
        fleet_vals_closed = fleet_vals + [fleet_vals[0]]

        fig = go.Figure()

        # Fleet baseline polygon
        fig.add_trace(
            go.Scatterpolar(
                r=fleet_vals_closed,
                theta=categories_closed,
                name="Fleet Baseline Average",
                line=dict(color="#94a3b8", dash="dot", width=2),
                fill="none",
            )
        )

        # Selected vehicle polygon
        fig.add_trace(
            go.Scatterpolar(
                r=vehicle_vals_closed,
                theta=categories_closed,
                name=f"Unit {audited_vid}",
                line=dict(color="#2563eb", width=2.5),
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.18)",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10, color="#64748b"),
                    gridcolor="#e2e8f0",
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color="#0f172a", family="Inter, sans-serif"),
                    gridcolor="#e2e8f0",
                ),
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=40, r=40, t=20, b=40),
            height=360,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_breakdown:
        st.markdown("**Explainable Deduction Audit Trail**")
        rep = radar_data["report"]
        st.markdown(
            f"""
            <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:12px; font-size:12px; line-height:1.6;">
                <div style="font-weight:700; font-size:14px; color:#0f172a; margin-bottom:8px;">
                    Unit {audited_vid} — Score Formulation ({selected_date.strftime('%Y-%m-%d')})
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding:4px 0;">
                    <span style="color:#64748b;">Baseline Starting Score:</span>
                    <span style="font-weight:600; color:#0f172a;">100.0 pts</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding:4px 0;">
                    <span style="color:#64748b;">Fuel Theft Deductions:</span>
                    <span style="font-weight:600; color:#dc2626;">-{rep.breakdown.get('fuel_theft', 0.0):.2f} pts</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding:4px 0;">
                    <span style="color:#64748b;">Route & Geofence Deductions:</span>
                    <span style="font-weight:600; color:#dc2626;">-{rep.breakdown.get('moonlighting', 0.0):.2f} pts</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding:4px 0;">
                    <span style="color:#64748b;">Speed & Kinematic Deductions:</span>
                    <span style="font-weight:600; color:#dc2626;">-{(rep.breakdown.get('speeding', 0.0) + rep.breakdown.get('harsh_driving', 0.0)):.2f} pts</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding:4px 0;">
                    <span style="color:#64748b;">Idling Wastage Deductions:</span>
                    <span style="font-weight:600; color:#d97706;">-{rep.breakdown.get('idling', 0.0):.2f} pts</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding:4px 0;">
                    <span style="color:#64748b;">Compounding Factor Applied:</span>
                    <span style="font-weight:600; color:{'#dc2626' if rep.compounding_applied else '#16a34a'};">
                        {'True (' + str(compound_mult) + 'x)' if rep.compounding_applied else 'None'}
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; padding-top:8px; font-size:13px;">
                    <span style="font-weight:700; color:#0f172a;">Calibrated Driver Safety Score:</span>
                    <span style="font-weight:700; color:{'#16a34a' if rep.safety_score >= 75.0 else '#dc2626'};">
                        {rep.safety_score:.2f} / 100
                    </span>
                </div>
                <div style="margin-top:4px; font-size:11px; color:#64748b;">
                    Risk Tier: <strong>{rep.risk_tier}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # 8. Audit Trail & CSV Export
    # --------------------------------------------------------------------------
    st.markdown("### Forensic Audit Trail & Enterprise Export")

    t_audit_tab, t_export_tab = st.tabs(["Historical Fuel Loss Incidents", "Shift Report Export"])

    with t_audit_tab:
        fuel_logs_df = service.get_fuel_loss_records(
            target_date=selected_date,
            shift_window=selected_shift,
            limit=200,
        )
        if not fuel_logs_df.empty:
            fl_view = fuel_logs_df[["id", "vehicle_id", "timestamp_myt", "incident_type", "magnitude_liters", "status"]].rename(
                columns={
                    "id": "Log ID",
                    "vehicle_id": "Vehicle ID",
                    "timestamp_myt": "Timestamp (MYT)",
                    "incident_type": "Incident Type",
                    "magnitude_liters": "Magnitude (L)",
                    "status": "Audit Status",
                }
            )
            st.dataframe(fl_view, use_container_width=True, hide_index=True)
        else:
            render_notice_banner(
                f"No fuel theft or sudden drop anomalies logged for {selected_date.strftime('%Y-%m-%d')} during '{selected_shift}'.",
                banner_type="success",
            )

    with t_export_tab:
        st.markdown(
            "Download the complete end-of-shift forensic leaderboard and risk classifications as a formatted CSV report for enterprise stakeholder audits."
        )

        csv_data = leaderboard_df.to_csv(index=False)
        filename_myt = f"fleet_forensic_audit_{selected_date.strftime('%Y%m%d')}_{now_myt().strftime('%H%M%S')}_MYT.csv"

        st.download_button(
            label="Download Complete Shift Audit Report (CSV)",
            data=csv_data,
            file_name=filename_myt,
            mime="text/csv",
            type="primary",
        )
