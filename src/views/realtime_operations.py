"""Workspace 1: Mode A — Live Operational Stream View.

Implements high-density live operational streaming, interactive Folium Leaflet map with OpenStreetMap tiles,
real-time kinematic telemetry strip, active unread incident queue with operational acknowledgment,
and Telegram Webhook integration with .env credential loading and automated incident triggers.
Strictly enforces Malaysia Standard Time (MYT, UTC+8) with zero UI emojis.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.components.ui import (
    get_freshness_badge_html,
    get_risk_tier_badge_html,
    render_empty_state,
    render_metric_card,
    render_notice_banner,
    render_page_header,
)
from src.services.data_service import TelematicsDataService
from src.utils.timezone import now_myt, to_myt_str

logger = logging.getLogger(__name__)

# Default map center for Malaysia operations (Kuala Lumpur / Klang Valley)
DEFAULT_CENTER_LAT = 3.1390
DEFAULT_CENTER_LON = 101.6869


def render_realtime_operations_view(
    service: TelematicsDataService,
    replay_snapshot_df: Optional[pd.DataFrame] = None,
    is_replay_mode: bool = False,
    replay_as_of_myt: Optional[str] = None,
) -> None:
    """Render Mode A: Live Operational Stream.

    Parameters
    ----------
    service : TelematicsDataService
        Data service layer instance.
    replay_snapshot_df : Optional[pd.DataFrame], optional
        Optional telemetry dataframe slice from the Replay Engine. Defaults to None.
    is_replay_mode : bool, optional
        Whether Replay Mode is active. Defaults to False.
    replay_as_of_myt : Optional[str], optional
        MYT timestamp string of the replay scrubber.
    """
    # --------------------------------------------------------------------------
    # 1. Operational Status Header & Metadata
    # --------------------------------------------------------------------------
    kpis = service.get_fleet_kpis()
    current_time_str = now_myt().strftime("%Y-%m-%d %H:%M:%S MYT (UTC+8)")

    if is_replay_mode and replay_as_of_myt:
        mode_badge = ("Operational Architecture", "Mode A: Historical Replay")
        clock_badge = ("Replay Cutoff", f"{replay_as_of_myt} MYT")
    else:
        mode_badge = ("Operational Architecture", "Mode A: Live Operational Stream")
        clock_badge = ("Live Clock", now_myt().strftime("%H:%M:%S MYT"))

    render_page_header(
        title="Mode A: Live Operational Stream",
        description="Active real-time operational feed, continuous multi-condition anomaly detection, active unread incident queue, and situational positioning.",
        meta_badges=[
            mode_badge,
            clock_badge,
            ("Active on Road", f"{kpis.get('active_vehicles', 0)} / {kpis.get('total_vehicles', 0)} Vehicles"),
            ("Unread Incident Queue", f"{kpis.get('unresolved_anomalies', 0)} Active Alerts"),
        ],
    )

    if is_replay_mode:
        render_notice_banner(
            f"Historical Telematics Replay active as of {replay_as_of_myt} MYT. Telemetry markers and kinematic metrics reflect historical state.",
            banner_type="info",
            badge="REPLAY MODE",
        )

    # --------------------------------------------------------------------------
    # 2. Prepare Fleet Telemetry Data (Live or Replayed)
    # --------------------------------------------------------------------------
    if replay_snapshot_df is not None and not replay_snapshot_df.empty:
        fleet_df = replay_snapshot_df
    else:
        fleet_df = service.get_fleet_status_table()

    if fleet_df.empty:
        render_empty_state(
            title="No Fleet Telemetry Available",
            description="The database currently contains zero registered vehicles or telemetry records.",
        )
        return

    # --------------------------------------------------------------------------
    # 3. Interactive Folium Geospatial Map
    # --------------------------------------------------------------------------
    st.markdown("### Fleet Geospatial Surveillance")

    # Filter records with valid latitude/longitude
    valid_coords = fleet_df.dropna(subset=["latitude", "longitude"])
    valid_coords = valid_coords[(valid_coords["latitude"] != 0.0) & (valid_coords["longitude"] != 0.0)]

    if not valid_coords.empty:
        center_lat = float(valid_coords["latitude"].mean())
        center_lon = float(valid_coords["longitude"].mean())
        initial_zoom = 11
    else:
        center_lat = DEFAULT_CENTER_LAT
        center_lon = DEFAULT_CENTER_LON
        initial_zoom = 10

    # Build Folium Leaflet Map with OpenStreetMap tiles explicitly
    fleet_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=initial_zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Pre-fetch recent anomalies to color-code vehicles dynamically
    anomaly_counts = service.get_anomaly_counts_by_vehicle()

    for _, row in valid_coords.iterrows():
        vid = str(row["vehicle_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        speed = float(row.get("speed") or 0.0)
        fuel = float(row.get("fuel_level") or 0.0)
        fuel_cap = float(row.get("fuel_capacity_liters") or 400.0)
        fuel_pct = min(100.0, max(0.0, (fuel / fuel_cap) * 100.0)) if fuel_cap > 0 else 0.0
        ignition = str(row.get("ignition_label") or "OFF")
        raw_ts = row.get("timestamp")
        ts_myt = to_myt_str(raw_ts)

        v_anoms = anomaly_counts.get(vid, {})
        has_critical = (
            v_anoms.get("fuel_theft", 0) > 0
            or v_anoms.get("moonlighting", 0) > 0
            or v_anoms.get("speeding", 0) > 0
        )

        # Dynamic color coding:
        # Red = Active High Risk / Fuel Theft / Geofence Breach
        # Orange/Amber = Stationary / Idling / Low Fuel (< 20%)
        # Green = Normal Operations
        if has_critical:
            marker_color = "red"
            status_text = "Security Alert Triggered"
            icon_name = "exclamation-sign"
        elif speed < 1.0 or fuel_pct < 20.0:
            marker_color = "orange"
            status_text = "Stationary / Low Fuel" if fuel_pct < 20.0 else "Stationary / Idle"
            icon_name = "time"
        else:
            marker_color = "green"
            status_text = "Normal Transit"
            icon_name = "ok-sign"

        popup_html = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif; min-width:190px; padding:2px;">
            <div style="font-weight:700; font-size:13px; color:#0f172a; margin-bottom:2px;">Vehicle {vid}</div>
            <div style="font-size:11px; margin-bottom:6px; color:{'#dc2626' if marker_color=='red' else ('#d97706' if marker_color=='orange' else '#16a34a')}; font-weight:600;">
                {status_text}
            </div>
            <div style="font-size:11px; line-height:1.6; color:#334155; border-top:1px solid #e2e8f0; padding-top:4px;">
                <strong>Speed:</strong> {speed:.1f} km/h<br/>
                <strong>Ignition:</strong> {ignition}<br/>
                <strong>Fuel Tank:</strong> {fuel:.1f} L ({fuel_pct:.0f}%)<br/>
                <strong>Last Ping (MYT):</strong> {ts_myt}
            </div>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"Unit {vid} — {speed:.0f} km/h — {fuel_pct:.0f}% Fuel",
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix="glyphicon"),
        ).add_to(fleet_map)

    # Render folium map in Streamlit container
    st_folium(fleet_map, width="100%", height=420, returned_objects=[])

    # --------------------------------------------------------------------------
    # 4. Kinematic Telemetry Strip for Selected Vehicle
    # --------------------------------------------------------------------------
    st.markdown("### Active Telemetry Signal Strip")

    available_vehicles = fleet_df["vehicle_id"].astype(str).tolist()
    c_unit, _ = st.columns([2, 3])
    with c_unit:
        selected_vid = st.selectbox(
            "Select Vehicle for Real-Time Telemetry Inspection",
            options=available_vehicles,
            index=0,
            help="Select an authentic Innonics vehicle ID from the active fleet registry.",
        )

    # Extract target vehicle state row
    v_rows = fleet_df[fleet_df["vehicle_id"].astype(str) == str(selected_vid)]
    if not v_rows.empty:
        v_data = v_rows.iloc[0]
        v_speed = float(v_data.get("speed") or 0.0)
        v_fuel = float(v_data.get("fuel_level") or 0.0)
        v_cap = float(v_data.get("fuel_capacity_liters") or 400.0)
        v_fuel_pct = (v_fuel / v_cap) * 100.0 if v_cap > 0 else 0.0
        v_ignition = str(v_data.get("ignition_label") or "OFF")
        v_dss = float(v_data.get("safety_score") or 100.0)
        v_tier = str(v_data.get("risk_tier") or "Excellent")
        v_ts_raw = v_data.get("timestamp")
        v_ts_myt = to_myt_str(v_ts_raw)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            render_metric_card(
                label="Current Speed",
                value=f"{v_speed:.1f} km/h",
                subtitle="Cellular GPS Doppler",
                delta="Nominal" if v_speed <= 90.0 else "Overspeed",
                delta_color="normal" if v_speed <= 90.0 else "inverse",
            )
        with m2:
            render_metric_card(
                label="Ignition & Engine",
                value=v_ignition,
                subtitle=f"Heading: {float(v_data.get('heading') or 0.0):.0f} deg",
                delta="Engine Running" if v_ignition == "ON" else "Parked / Off",
                delta_color="normal" if v_ignition == "ON" else "off",
            )
        with m3:
            fuel_delta = f"{v_fuel_pct:.1f}% Tank"
            fuel_color = "normal" if v_fuel_pct >= 20.0 else "inverse"
            render_metric_card(
                label="Fuel Tank Level",
                value=f"{v_fuel:.1f} L",
                subtitle=f"Capacity: {v_cap:.0f} L",
                delta=fuel_delta,
                delta_color=fuel_color,
            )
        with m4:
            tier_color = "normal" if v_dss >= 75.0 else "inverse"
            render_metric_card(
                label="Driver Safety Score",
                value=f"{v_dss:.1f} / 100",
                subtitle=v_tier,
                delta="Compliant" if v_dss >= 75.0 else "Elevated Risk",
                delta_color=tier_color,
            )
        with m5:
            render_metric_card(
                label="Telemetry Timestamp",
                value=v_ts_myt.split(" ")[-1] if " " in v_ts_myt else v_ts_myt,
                subtitle=f"{v_ts_myt.split(' ')[0]} MYT",
                delta="UTC+8 Synchronized",
                delta_color="normal",
            )

    # --------------------------------------------------------------------------
    # 5. Real-Time Push Log & Alert Simulation
    # --------------------------------------------------------------------------
    st.markdown("### Operational Event Logs & Alert Dispatch")

    tab_pings, tab_alerts, tab_webhook = st.tabs([
        "Live Telemetry Stream",
        "Active Incident Queue",
        "Telegram Webhook Integration",
    ])

    with tab_pings:
        pings_df = service.get_vehicle_telemetry(vehicle_id=selected_vid, limit=100)
        if not pings_df.empty:
            pings_view = pings_df[["timestamp", "speed", "fuel_level", "ignition", "latitude", "longitude"]].copy()
            pings_view["timestamp_myt"] = pings_view["timestamp"].apply(to_myt_str)
            pings_view["ignition"] = pings_view["ignition"].apply(lambda v: "ON" if str(v) in ("1", "1.0", "ON") else "OFF")
            pings_view = pings_view.rename(
                columns={
                    "timestamp_myt": "Timestamp (MYT)",
                    "speed": "Speed (km/h)",
                    "fuel_level": "Fuel (L)",
                    "ignition": "Ignition",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                }
            )[["Timestamp (MYT)", "Speed (km/h)", "Fuel (L)", "Ignition", "Latitude", "Longitude"]]

            st.dataframe(
                pings_view.iloc[::-1].head(25),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_empty_state("No Pings Logged", f"No telemetry transmissions found for unit {selected_vid}.")

    with tab_alerts:
        c_filter1, c_filter2 = st.columns([2, 3])
        with c_filter1:
            queue_scope = st.radio(
                "Incident Queue Scope",
                options=["Unread / Open Alerts Only", "All Historical Incidents"],
                index=0,
                horizontal=True,
            )

        status_filter = "OPEN" if queue_scope == "Unread / Open Alerts Only" else None
        alerts_df = service.get_anomalies(vehicle_id=selected_vid, status=status_filter, limit=100)

        if not alerts_df.empty:
            alerts_view = alerts_df[["id", "vehicle_id", "timestamp", "incident_type", "magnitude", "status"]].copy()
            alerts_view["timestamp_myt"] = alerts_view["timestamp"].apply(to_myt_str)
            alerts_view = alerts_view.rename(
                columns={
                    "id": "Incident ID",
                    "vehicle_id": "Vehicle ID",
                    "timestamp_myt": "Timestamp (MYT)",
                    "incident_type": "Incident Type",
                    "magnitude": "Magnitude",
                    "status": "Queue Status",
                }
            )[["Incident ID", "Vehicle ID", "Timestamp (MYT)", "Incident Type", "Magnitude", "Queue Status"]]

            st.dataframe(
                alerts_view.head(25),
                use_container_width=True,
                hide_index=True,
            )

            # Interactive Acknowledgment / Resolution Action Form
            open_alerts = alerts_df[alerts_df["status"].str.upper() != "RESOLVED"]
            if not open_alerts.empty:
                with st.expander("Acknowledge & Resolve Incident (Audit Justification)", expanded=False):
                    c_act1, c_act2 = st.columns([2, 3])
                    with c_act1:
                        target_incident_id = st.selectbox(
                            "Select Incident to Acknowledge",
                            options=open_alerts["id"].tolist(),
                            format_func=lambda x: f"Incident #{x} ({open_alerts.loc[open_alerts['id'] == x, 'incident_type'].values[0]})",
                        )
                        res_justification = st.text_input(
                            "Resolution Justification / Audit Action Taken",
                            placeholder="e.g. Verified with driver; maintenance inspected fuel tank.",
                        )
                        btn_resolve = st.button("Mark Incident Resolved", type="primary")

                    if btn_resolve:
                        if not res_justification.strip():
                            st.error("Please enter a resolution justification rationale.")
                        else:
                            ok, msg = service.resolve_anomaly(
                                anomaly_id=int(target_incident_id),
                                resolution_notes=res_justification.strip(),
                                user_id="ops_dispatcher",
                            )
                            if ok:
                                st.success(f"Incident #{target_incident_id} successfully acknowledged and marked as RESOLVED.")
                                st.rerun()
                            else:
                                st.error(f"Resolution failed: {msg}")
        else:
            render_notice_banner(
                f"Vehicle {selected_vid} currently has zero active unread incidents in the queue.",
                banner_type="success",
                title="Zero Active Anomalies",
            )

    with tab_webhook:
        tg_svc = service.telegram_service
        st.markdown(
            "Enterprise Telegram Bot Webhook Gateway for immediate operational escalations upon critical anomaly detection. "
            "Credentials are loaded securely from the `.env` environment file."
        )

        # Gateway Status Card
        gateway_mode_str = "LIVE TELEGRAM BOT API" if tg_svc.is_configured else "VERIFIED LOCAL SIMULATION GATEWAY"
        status_color = "#16a34a" if tg_svc.is_configured else "#2563eb"
        st.markdown(
            f"""
            <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:12px 16px; margin-bottom:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:700; font-size:13px; color:#0f172a;">Gateway Operational Status</div>
                        <div style="font-size:12px; color:{status_color}; font-weight:600; margin-top:2px;">
                            {gateway_mode_str}
                        </div>
                    </div>
                    <div style="text-align:right; font-size:11px; color:#64748b;">
                        <div>Config Source: <code>.env</code></div>
                        <div>Token: <code>{tg_svc.get_masked_token()}</code></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_w1, c_w2 = st.columns([2, 3])
        with c_w1:
            st.markdown("**Manual Verification Ping (Manual Dispatch)**")
            test_chat_id = st.text_input("Target Chat / Channel ID", value=tg_svc.chat_id)
            btn_test_ping = st.button("Dispatch Manual Verification Ping", use_container_width=True)

            st.markdown("---")
            st.markdown("**Automated Anomaly Escalation Trigger**")
            sim_vehicle = st.selectbox(
                "Trigger Vehicle ID",
                options=available_vehicles,
                index=available_vehicles.index(selected_vid) if selected_vid in available_vehicles else 0,
            )
            sim_type = st.selectbox(
                "Critical Incident Type",
                options=["FUEL_THEFT", "UNAUTHORIZED_NIGHT_DRIVING", "GEOFENCE_BREACH", "OVERSPEEDING"],
                index=0,
            )
            sim_mag = st.number_input(
                "Incident Magnitude",
                value=-3.50,
                step=0.5,
                format="%.2f",
                help="Magnitude in Liters (for fuel) or km/h (for speed).",
            )
            btn_trigger_alert = st.button("Trigger Critical Anomaly Webhook", type="primary", use_container_width=True)

        with c_w2:
            st.markdown("**Payload Preview (Corporate Markdown Format)**")
            preview_payload = tg_svc.format_incident_payload(
                vehicle_id=sim_vehicle,
                incident_type=sim_type,
                magnitude=sim_mag,
            )
            st.code(preview_payload, language="text")

            if btn_test_ping:
                test_receipt = tg_svc.send_test_alert(chat_id=test_chat_id)
                st.success(
                    f"Manual verification ping dispatched at {test_receipt.get('timestamp_myt')} "
                    f"(HTTP {test_receipt.get('http_status')}, Mode: {test_receipt.get('mode')})."
                )
                st.json(test_receipt)

            if btn_trigger_alert:
                alert_receipt = tg_svc.trigger_anomaly_alert(
                    vehicle_id=sim_vehicle,
                    incident_type=sim_type,
                    magnitude=sim_mag,
                )
                st.success(
                    f"Critical anomaly alert dispatched to Telegram at {alert_receipt.get('timestamp_myt')} "
                    f"(HTTP {alert_receipt.get('http_status')}, Mode: {alert_receipt.get('mode')})."
                )
                st.json(alert_receipt)
