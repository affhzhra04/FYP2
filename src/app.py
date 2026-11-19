"""Main Application Entry Point for Enterprise Fleet Decision Support Platform.

Dual-Mode Commercial Telematics Architecture:
  - Workspace 1: Real-Time Fleet Operations (Geospatial Folium map, telemetry strip, incident feed)
  - Workspace 2: Forensic Audit & MCDA Decision Support (Shift leaderboard, dynamic MCDA sliders, XAI radar)
  - Enterprise Control Bar: Telematics Stream & Replay Engine
Strictly enforces Malaysia Standard Time (MYT, UTC+8).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Optional
from zoneinfo import ZoneInfo

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.components.ui import (
    inject_custom_css,
    render_empty_state,
    render_notice_banner,
)
from src.services.data_service import TelematicsDataService, get_data_service
from src.utils.timezone import (
    MYT_TIMEZONE_STR,
    MYT_TZ,
    now_myt,
    to_myt_datetime,
    to_myt_str,
)
from src.views import (
    render_fleet_config_page,
    render_forensic_audit_view,
    render_realtime_operations_view,
)

# Configure enterprise logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fleet_operation_dashboard")


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="Enterprise Fleet Decision Support Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject enterprise CSS design system
inject_custom_css()


# ==============================================================================
# Main Application Controller
# ==============================================================================

def main() -> None:
    """Main application lifecycle controller and workspace router."""
    service = get_data_service()

    # --------------------------------------------------------------------------
    # Sidebar Navigation & Telematics Stream / Replay Engine
    # --------------------------------------------------------------------------
    with st.sidebar:
        brand_html = (
            '<div class="sidebar-brand-box">'
            '<div class="sidebar-brand-title">'
            '<span>Fleet Operations</span>'
            '<span class="sidebar-live-pill">LIVE</span>'
            '</div>'
            '<div class="sidebar-brand-sub">Enterprise Telematics Platform</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)

        # Primary Workspace Selector (Dual-Mode Architecture)
        workspace_options = [
            "Mode A: Live Operational Stream",
            "Mode B: Forensic Audit Mode",
            "Fleet Registry & Schedules",
        ]

        selected_workspace = st.radio(
            "Enterprise Workspaces",
            options=workspace_options,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Telematics Stream & Replay Engine (Enterprise Control Bar)
        # ----------------------------------------------------------------------
        st.markdown("##### Telematics Stream & Replay Engine")

        # 1. Data Source Selector
        data_source = st.radio(
            "Data Ingestion Source",
            options=[
                "Live Cellular MQTT Broker (Production)",
                "Historical Telematics Replay (Audit Mode)",
            ],
            index=0,
            help="Switch between live production MQTT broker ingest and historical forensic telemetry replay.",
        )

        is_replay_mode = "Historical Telematics Replay" in data_source
        replay_snapshot_df: Optional[pd.DataFrame] = None
        replay_as_of_myt: Optional[str] = None

        # 2. Replay Speed Slider
        replay_speed = st.select_slider(
            "Replay Playback Speed",
            options=["1x (Real-Time)", "5x", "10x", "30x (Fast Audit)"],
            value="1x (Real-Time)",
            help="Simulation clock acceleration factor for fast forensic playback.",
        )

        # 3. Timeline Scrubber (Active when Replay Mode is selected)
        if is_replay_mode:
            st.markdown("**Timeline Scrubber (MYT, UTC+8)**")

            # Query historical temporal boundaries from SQLite
            with service.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM telematics_pings;")
                row = cursor.fetchone()
                db_min_ts, db_max_ts = (row[0], row[1]) if row else (None, None)

            if db_min_ts and db_max_ts:
                min_dt_myt = to_myt_datetime(db_min_ts).replace(tzinfo=None)
                max_dt_myt = to_myt_datetime(db_max_ts).replace(tzinfo=None)

                preset_window = st.selectbox(
                    "Shift Window Presets",
                    options=[
                        "August 2024 Shift Window (Full Range)",
                        "Ghost Hours Window (22:00 - 04:59 MYT)",
                        "Day Operations Window (08:00 - 17:00 MYT)",
                        "Custom Timeline Scrub Point",
                    ],
                    index=0,
                )

                if "Ghost Hours" in preset_window:
                    # Target a ghost hours window point
                    scrub_target = min_dt_myt.replace(hour=23, minute=15, second=0)
                    if scrub_target > max_dt_myt or scrub_target < min_dt_myt:
                        scrub_target = min_dt_myt
                elif "Day Operations" in preset_window:
                    # Target standard daytime shift point
                    scrub_target = min_dt_myt.replace(hour=14, minute=30, second=0)
                    if scrub_target > max_dt_myt or scrub_target < min_dt_myt:
                        scrub_target = min_dt_myt
                else:
                    scrub_target = max_dt_myt

                scrubbed_dt = st.slider(
                    "Scrub Timeline (MYT)",
                    min_value=min_dt_myt,
                    max_value=max_dt_myt,
                    value=scrub_target if min_dt_myt <= scrub_target <= max_dt_myt else max_dt_myt,
                    format="YYYY-MM-DD HH:mm",
                    help="Jump to a specific historical shift timestamp (MYT, UTC+8).",
                )

                # Convert scrubbed naive MYT back to UTC for database querying
                scrubbed_myt_aware = scrubbed_dt.replace(tzinfo=MYT_TZ)
                utc_cutoff = scrubbed_myt_aware.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                replay_snapshot_df = service.get_replay_telemetry_snapshot(as_of_timestamp=utc_cutoff)
                replay_as_of_myt = scrubbed_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.caption("No historical telemetry records found in database to scrub.")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Operations Metadata & Polling Rate
        # ----------------------------------------------------------------------
        ops_html = (
            f'<div class="sidebar-ops-card">'
            f'<div class="sidebar-ops-row"><span>Timezone</span><span class="sidebar-ops-val" style="color:#0ea5e9;">MYT (UTC+8)</span></div>'
            f'<div class="sidebar-ops-row"><span>Clock</span><span class="sidebar-ops-val">{now_myt().strftime("%H:%M:%S")}</span></div>'
            f'<div class="sidebar-ops-row"><span>Status</span><span class="sidebar-ops-val" style="color:#22c55e;">Operational</span></div>'
            f'<div class="sidebar-ops-row"><span>Speed</span><span class="sidebar-ops-val">{replay_speed}</span></div>'
            f'</div>'
        )
        st.markdown(ops_html, unsafe_allow_html=True)

        c_ref1, c_ref2 = st.columns([1.2, 1])
        with c_ref1:
            refresh_interval = st.selectbox(
                "Polling Rate",
                options=["Manual", "15s", "30s", "60s"],
                index=0,
                label_visibility="collapsed",
                help="Automated telemetry polling interval",
            )
        with c_ref2:
            if st.button("Refresh", use_container_width=True, type="primary"):
                st.cache_data.clear()
                st.rerun()

    # --------------------------------------------------------------------------
    # Dispatch Selected Workspace View
    # --------------------------------------------------------------------------
    try:
        if "Mode A" in selected_workspace or "Live Operational" in selected_workspace or "Real-Time" in selected_workspace:
            render_realtime_operations_view(
                service=service,
                replay_snapshot_df=replay_snapshot_df,
                is_replay_mode=is_replay_mode,
                replay_as_of_myt=replay_as_of_myt,
            )
        elif "Mode B" in selected_workspace or "Forensic Audit" in selected_workspace:
            render_forensic_audit_view(service=service)
        elif "Fleet Registry" in selected_workspace:
            render_fleet_config_page(service=service)
        else:
            render_empty_state("Invalid View", "The requested workspace perspective does not exist.")
    except Exception as exc:
        logger.exception("Dashboard view execution failure: %s", exc)
        st.error(f"Operational error: Unable to render selected workspace ({exc}).")


if __name__ == "__main__":
    main()
