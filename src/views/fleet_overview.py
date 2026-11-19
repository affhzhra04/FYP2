"""Fleet Operational Overview & Monitor View.

Enterprise-grade situational monitoring, fleet connectivity status grid,
priority operational incident queue, and telemetry risk analytics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.components.ui import (
    render_empty_state,
    render_metric_card,
    render_page_header,
)
from src.services.data_service import TelematicsDataService


def render_fleet_overview_page(service: TelematicsDataService) -> None:
    """Render Page: Fleet Operational Overview & Monitor."""
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # 1. Compact Page Header
    render_page_header(
        title="Fleet Operational Overview & Monitor",
        description="Real-time situational monitoring, vehicle connectivity statuses, and operational incident queue.",
        meta_badges=[
            ("System", "Operational"),
            ("Data Window", "24h Window"),
            ("Last Sync", now_str),
        ],
    )

    freshness_hours = 24.0
    freshness_mode = "relative_to_dataset"

    # 2. Executive KPI Cards (Single Consistent Row)
    kpis = service.get_fleet_kpis(
        freshness_hours=freshness_hours,
        freshness_mode=freshness_mode,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card(
            label="Total Fleet Units",
            value=f"{kpis['total_vehicles']}",
            subtitle="Commercial fleet assets",
        )
    with m2:
        render_metric_card(
            label="Active Units (<24h)",
            value=f"{kpis['active_vehicles']} / {kpis['total_vehicles']}",
            subtitle="Reporting telemetry",
            delta=f"{kpis['active_vehicles']} Online",
            delta_color="normal",
        )
    with m3:
        render_metric_card(
            label="Total Telemetry Pings",
            value=f"{kpis['total_pings']:,}",
            subtitle="Persisted sensor frames",
        )
    with m4:
        render_metric_card(
            label="Unresolved Incidents",
            value=f"{kpis['unresolved_anomalies']}",
            subtitle="Requires operator review",
            delta=f"{kpis['unresolved_anomalies']} Open" if kpis['unresolved_anomalies'] > 0 else "0 Open",
            delta_color="inverse" if kpis['unresolved_anomalies'] > 0 else "normal",
        )
    with m5:
        score_val = kpis["average_safety_score"]
        render_metric_card(
            label="Fleet Safety Score",
            value=f"{score_val:.1f} / 100",
            subtitle=f"Average across {kpis['scored_vehicles_count']} vehicles",
        )

    # Query fleet status table
    fleet_table = service.get_fleet_status_table(
        freshness_hours=freshness_hours,
        freshness_mode=freshness_mode,
    )

    def format_incident_category(cat_str: str) -> str:
        mapping = {
            "SPEEDING": "Speeding",
            "OVERSPEEDING": "Speeding",
            "HARSH_BRAKING": "Harsh Braking",
            "HARSH_ACCELERATION": "Harsh Acceleration",
            "HARSH_CORNERING": "Harsh Cornering",
            "EXCESSIVE_IDLING": "Excessive Idling",
            "FUEL_THEFT": "Fuel Theft",
            "UNAUTHORIZED_NIGHT_DRIVING": "Night Shift Violation",
            "GEOFENCE_BREACH": "Geofence Breach",
            "MOONLIGHTING": "Unauthorized Trip",
        }
        return mapping.get(str(cat_str).upper(), str(cat_str).replace("_", " ").title())

    # 3. Two-Column Primary Monitoring Section: Fleet Table (Left) & Incident Queue (Right)
    col_fleet, col_incidents = st.columns([1.55, 1.45], gap="medium")

    # --------------------------------------------------------------------------
    # Left Column: Fleet Registry & Connectivity Status
    # --------------------------------------------------------------------------
    with col_fleet:
        st.markdown("### Fleet Telemetry & Connectivity Status")

        if fleet_table.empty:
            render_empty_state(
                title="No Fleet Vehicles Configured",
                description="The fleet registry currently contains zero vehicles. Add vehicles in Fleet Configuration.",
            )
        else:
            # Compact Inline Filters
            f1, f2, f3 = st.columns([1.2, 0.9, 0.9])
            with f1:
                search_vid = st.text_input(
                    "Search Vehicle ID:",
                    placeholder="Search Vehicle ID...",
                    key="ov_search_vid",
                    label_visibility="collapsed",
                )
            with f2:
                status_filter = st.selectbox(
                    "Connectivity Status:",
                    options=["All", "Active", "Recent", "Stale", "No Telemetry"],
                    index=0,
                    key="ov_status_filt",
                    label_visibility="collapsed",
                )
            with f3:
                tier_options = ["All", "Critical", "High", "Moderate", "Low", "Excellent"]
                tier_filter = st.selectbox(
                    "Risk Tier Filter:",
                    options=tier_options,
                    index=0,
                    key="ov_tier_filt",
                    label_visibility="collapsed",
                )

            # Apply filtering
            filtered_fleet = fleet_table.copy()
            if status_filter != "All":
                filtered_fleet = filtered_fleet[filtered_fleet["freshness_status"] == status_filter]
            if tier_filter != "All":
                filtered_fleet = filtered_fleet[
                    filtered_fleet["risk_tier"].astype(str).str.contains(tier_filter, case=False, na=False)
                ]
            if search_vid.strip():
                sq = search_vid.strip().lower()
                filtered_fleet = filtered_fleet[filtered_fleet["vehicle_id"].str.lower().str.contains(sq, na=False)]

            if filtered_fleet.empty:
                render_empty_state(
                    title="No Matching Fleet Records",
                    description="No vehicle records match the selected status or risk tier filters.",
                )
            else:
                display_df = pd.DataFrame()
                display_df["Vehicle ID"] = filtered_fleet["vehicle_id"]
                display_df["Status"] = filtered_fleet["freshness_status"]
                display_df["Safety Score"] = filtered_fleet["safety_score"].apply(
                    lambda s: f"{s:.1f}" if pd.notna(s) else "N/A"
                )
                display_df["Risk Tier"] = filtered_fleet["risk_tier"].apply(
                    lambda t: t.value.split(" ")[0] if hasattr(t, "value") else str(t).split(" ")[0]
                )
                display_df["Speed (km/h)"] = filtered_fleet["speed"].apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "0.0"
                )
                display_df["Fuel Level (L)"] = filtered_fleet["fuel_level"].apply(
                    lambda f: f"{f:.1f}" if pd.notna(f) else "—"
                )
                display_df["Ignition"] = filtered_fleet["ignition_label"]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )

    # --------------------------------------------------------------------------
    # Right Column: Priority Incident Queue & Actions
    # --------------------------------------------------------------------------
    with col_incidents:
        st.markdown("### Priority Operational Incident Queue")

        # Compact Queue Filters
        q1, q2 = st.columns([1, 1])
        with q1:
            inc_status = st.selectbox(
                "Queue Status:",
                options=["Open", "Resolved", "All"],
                index=0,
                key="ov_queue_status",
                label_visibility="collapsed",
            )
        with q2:
            inc_cat = st.selectbox(
                "Incident Category:",
                options=[
                    "All",
                    "Speeding",
                    "Harsh Braking",
                    "Harsh Acceleration",
                    "Harsh Cornering",
                    "Excessive Idling",
                    "Fuel Theft",
                    "Geofence Breach",
                    "Night Shift Violation",
                ],
                index=0,
                key="ov_queue_cat",
                label_visibility="collapsed",
            )

        cat_query = None
        if inc_cat != "All":
            inv_map = {
                "Speeding": "SPEEDING",
                "Harsh Braking": "HARSH_BRAKING",
                "Harsh Acceleration": "HARSH_ACCELERATION",
                "Harsh Cornering": "HARSH_CORNERING",
                "Excessive Idling": "EXCESSIVE_IDLING",
                "Fuel Theft": "FUEL_THEFT",
                "Geofence Breach": "GEOFENCE_BREACH",
                "Night Shift Violation": "UNAUTHORIZED_NIGHT_DRIVING",
            }
            cat_query = inv_map.get(inc_cat)

        anomalies_df = service.get_anomalies(
            vehicle_id=None,
            status=inc_status.upper() if inc_status != "All" else None,
            incident_type=cat_query,
            limit=50,
        )

        if anomalies_df.empty:
            render_empty_state(
                title="Incident Queue Clear",
                description="No active incidents match the current queue criteria.",
            )
        else:
            inc_display = pd.DataFrame()
            inc_display["ID"] = anomalies_df["id"]
            inc_display["Timestamp"] = anomalies_df["timestamp"].apply(lambda t: str(t)[:16])
            inc_display["Vehicle"] = anomalies_df["vehicle_id"]
            inc_display["Category"] = anomalies_df["incident_type"].apply(format_incident_category)
            inc_display["Magnitude"] = anomalies_df["magnitude"].apply(
                lambda m: f"{m:.2f}" if pd.notna(m) else "—"
            )
            inc_display["Status"] = anomalies_df["status"]

            st.dataframe(
                inc_display,
                use_container_width=True,
                hide_index=True,
                height=260,
            )

            # Compact Incident Resolution Action
            with st.expander("Supervisor Incident Sign-Off & Audit Form", expanded=False):
                open_incidents = anomalies_df[anomalies_df["status"] != "RESOLVED"]
                if not open_incidents.empty:
                    with st.form("form_resolve_incident_action"):
                        r1, r2 = st.columns([1, 1])
                        with r1:
                            target_id = st.selectbox(
                                "Target Incident ID:",
                                options=open_incidents["id"].tolist(),
                                format_func=lambda i: f"#{i} ({open_incidents[open_incidents['id'] == i]['vehicle_id'].values[0]} - {open_incidents[open_incidents['id'] == i]['incident_type'].values[0]})",
                            )
                        with r2:
                            operator_id = st.text_input("Reviewer Supervisor ID:", value="Ops_Desk_1")

                        resolution_reason = st.text_input(
                            "Mandatory Operational Justification:",
                            placeholder="e.g. Authorized route bypass, verified road condition variance.",
                        )

                        submit_resolve = st.form_submit_button("Sign Off & Resolve Incident", type="primary")

                        if submit_resolve:
                            if not resolution_reason.strip():
                                st.error("Operational justification rationale is required before resolving.")
                            else:
                                success, msg = service.resolve_anomaly(
                                    anomaly_id=int(target_id),
                                    resolution_notes=resolution_reason.strip(),
                                    user_id=operator_id.strip(),
                                )
                                if success:
                                    st.success(f"Incident #{target_id} marked as RESOLVED.")
                                    st.rerun()
                                else:
                                    st.error(msg)
                else:
                    st.info("All incidents in the currently displayed view are resolved.")

    # 4. Compact Visualizations Section: Fleet Activity & Risk Distribution
    st.markdown("### Operational Telemetry & Risk Analytics")
    c_viz1, c_viz2 = st.columns([1.1, 1.9], gap="medium")

    with c_viz1:
        if not fleet_table.empty and "risk_tier" in fleet_table.columns:
            tier_order = ["Critical", "High", "Moderate", "Low", "Excellent"]
            tier_colors = {
                "Critical": "#dc2626",
                "High": "#ea580c",
                "Moderate": "#d97706",
                "Low": "#2563eb",
                "Excellent": "#16a34a",
            }
            tier_series = fleet_table["risk_tier"].apply(
                lambda t: t.value.split(" ")[0] if hasattr(t, "value") else str(t).split(" ")[0]
            ).value_counts()

            tier_df = pd.DataFrame([
                {"Tier": t, "Count": int(tier_series.get(t, 0)), "Color": tier_colors.get(t, "#64748b")}
                for t in tier_order
            ])

            fig_tier = go.Figure(go.Bar(
                x=tier_df["Tier"],
                y=tier_df["Count"],
                marker_color=tier_df["Color"],
                text=tier_df["Count"],
                textposition="outside",
            ))
            fig_tier.update_layout(
                title=dict(text="Fleet Risk Tier Distribution", font=dict(size=12, color="#0f172a")),
                margin=dict(l=10, r=10, t=28, b=10),
                height=220,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#334155")),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
            )
            st.plotly_chart(fig_tier, use_container_width=True)
        else:
            render_empty_state("No Risk Data", "Fleet data not available for risk tier chart.")

    with c_viz2:
        all_anomalies = service.get_anomalies(limit=200)
        if not all_anomalies.empty and "incident_type" in all_anomalies.columns:
            cat_counts = all_anomalies["incident_type"].apply(format_incident_category).value_counts().reset_index()
            cat_counts.columns = ["Category", "Occurrences"]
            cat_counts = cat_counts.sort_values(by="Occurrences", ascending=True)

            fig_cat = go.Figure(go.Bar(
                x=cat_counts["Occurrences"],
                y=cat_counts["Category"],
                orientation="h",
                marker_color="#1e293b",
                text=cat_counts["Occurrences"],
                textposition="outside",
            ))
            fig_cat.update_layout(
                title=dict(text="Logged Infractions by Incident Category", font=dict(size=12, color="#0f172a")),
                margin=dict(l=10, r=15, t=28, b=10),
                height=220,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
                yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#334155")),
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            render_empty_state("No Incident Records", "No anomalies logged across fleet.")

    # 5. Geospatial Tracking in Collapsible Container
    with st.expander("Geospatial Fleet Map & Asset Spatial Distribution", expanded=False):
        map_data = fleet_table.copy() if not fleet_table.empty else pd.DataFrame()
        if not map_data.empty:
            valid_coords = map_data.dropna(subset=["latitude", "longitude"]).copy()
            valid_coords = valid_coords[
                (valid_coords["latitude"].between(-90, 90))
                & (valid_coords["longitude"].between(-180, 180))
                & (valid_coords["latitude"] != 0)
                & (valid_coords["longitude"] != 0)
            ]
        else:
            valid_coords = pd.DataFrame()

        if valid_coords.empty:
            render_empty_state(
                title="Geospatial Telemetry Unavailable",
                description="No valid coordinates found in the active fleet registry.",
            )
        else:
            def get_node_color(tier_val: Any) -> List[int]:
                t = str(tier_val).lower()
                if "critical" in t:
                    return [220, 38, 38, 220]
                elif "high" in t:
                    return [234, 88, 12, 220]
                elif "moderate" in t:
                    return [217, 119, 6, 220]
                elif "low" in t:
                    return [29, 78, 216, 220]
                return [22, 163, 74, 220]

            valid_coords["color"] = valid_coords["risk_tier"].apply(get_node_color)
            valid_coords["score_display"] = valid_coords["safety_score"].apply(lambda s: f"{s:.1f}" if pd.notna(s) else "N/A")
            valid_coords["speed_display"] = valid_coords["speed"].apply(lambda s: f"{s:.1f} km/h" if pd.notna(s) else "0.0 km/h")
            valid_coords["fuel_display"] = valid_coords["fuel_level"].apply(lambda f: f"{f:.1f} L" if pd.notna(f) else "—")

            center_lat = float(valid_coords["latitude"].median())
            center_lon = float(valid_coords["longitude"].median())

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10,
                pitch=0,
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=valid_coords,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius=220,
                pickable=True,
                auto_highlight=True,
            )

            tooltip = {
                "html": (
                    "<b>Vehicle ID:</b> {vehicle_id}<br/>"
                    "<b>Status:</b> {freshness_status}<br/>"
                    "<b>Safety Score:</b> {score_display}<br/>"
                    "<b>Speed:</b> {speed_display}<br/>"
                    "<b>Fuel Level:</b> {fuel_display}<br/>"
                    "<b>Ignition:</b> {ignition_label}"
                ),
                "style": {
                    "backgroundColor": "#0f172a",
                    "color": "#f8fafc",
                    "fontSize": "11px",
                    "borderRadius": "4px",
                    "padding": "6px 8px",
                },
            }

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="road",
                ),
                use_container_width=True,
            )
