"""Fleet Configuration & Incident Administration View.

Provides vehicle registry management, driver shift roster configuration with
night-shift exemptions, incident resolution management, and administrative audit trails.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import render_empty_state, render_page_header
from src.services.data_service import TelematicsDataService


def render_fleet_config_page(service: TelematicsDataService) -> None:
    """Render Page: Fleet Configuration & Administration."""
    render_page_header(
        title="Fleet Configuration & Administration",
        description="Commercial vehicle registry, driver shift schedules, incident review, and administrative audit trail management.",
        meta_badges=[
            ("Database Engine", "SQLite Core"),
            ("Audit Logging", "Active"),
        ],
    )

    tabs = st.tabs([
        "Vehicle Registry",
        "Driver Shift Rosters",
        "Incident Review",
        "Administrative Audit Trail",
    ])

    # ==========================================================================
    # Tab 1: Vehicle Registry
    # ==========================================================================
    with tabs[0]:
        st.subheader("Vehicle Registry Management")
        st.caption("Manage registered fleet assets in the database.")

        vehicles = service.db.get_all_vehicles()
        v_df = pd.DataFrame(vehicles) if vehicles else pd.DataFrame()

        # Display Existing Vehicles
        if not v_df.empty:
            st.dataframe(
                v_df[["vehicle_id", "plate_number", "model", "fuel_capacity_liters"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_empty_state(title="No Vehicles Registered")

        st.markdown("---")
        vc1, vc2 = st.columns(2)

        # Register New Vehicle Form
        with vc1:
            st.markdown("##### Register New Vehicle")
            with st.form("form_register_vehicle", clear_on_submit=True):
                new_vid = st.text_input("Vehicle ID:", placeholder="e.g. TRK-101 or 1_12_0").strip()
                new_plate = st.text_input("License Plate:", placeholder="e.g. WXY 4567").strip()
                new_model = st.text_input("Vehicle Model / Specification:", placeholder="e.g. Freightliner Cascadia").strip()
                new_fuel = st.number_input("Fuel Tank Capacity (Liters):", min_value=1.0, max_value=2000.0, value=300.0, step=10.0)

                submitted_reg = st.form_submit_button("Register Vehicle", type="primary")
                if submitted_reg:
                    success, msg = service.register_vehicle(
                        vehicle_id=new_vid,
                        plate_number=new_plate,
                        model=new_model,
                        fuel_capacity_liters=float(new_fuel),
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        # Edit Existing Vehicle Form
        with vc2:
            st.markdown("##### Edit or Delete Vehicle")
            if not v_df.empty:
                selected_edit_vid = st.selectbox(
                    "Select Vehicle to Edit/Delete:",
                    options=v_df["vehicle_id"].tolist(),
                    key="select_edit_vid",
                )
                curr_row = v_df[v_df["vehicle_id"] == selected_edit_vid].iloc[0]

                with st.form("form_edit_vehicle"):
                    edit_plate = st.text_input("Plate Number:", value=str(curr_row["plate_number"]))
                    edit_model = st.text_input("Model:", value=str(curr_row["model"]))
                    edit_fuel = st.number_input(
                        "Fuel Capacity (L):",
                        min_value=1.0,
                        max_value=2000.0,
                        value=float(curr_row["fuel_capacity_liters"]),
                        step=10.0,
                    )
                    submitted_edit = st.form_submit_button("Update Vehicle Details")

                    if submitted_edit:
                        success, msg = service.update_vehicle(
                            vehicle_id=selected_edit_vid,
                            plate_number=edit_plate,
                            model=edit_model,
                            fuel_capacity_liters=float(edit_fuel),
                        )
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                # Delete Vehicle with confirmation
                st.markdown("###### Consequential Action: Delete Vehicle")
                confirm_del = st.checkbox(f"Confirm deletion of vehicle '{selected_edit_vid}' and cascaded records", key="del_veh_chk")
                if st.button("Delete Vehicle", type="secondary", disabled=not confirm_del):
                    success, msg = service.delete_vehicle(vehicle_id=selected_edit_vid)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # ==========================================================================
    # Tab 2: Driver Shift Rosters
    # ==========================================================================
    with tabs[1]:
        st.subheader("Driver Shift Rosters & Night-Shift Authorization")
        st.caption(
            "Configure authorized driver operating shifts. Authorized night shifts exempt drivers "
            "from late-night shift breach alerts."
        )

        schedules = service.db.get_driver_schedules()
        s_df = pd.DataFrame(schedules) if schedules else pd.DataFrame()

        if not s_df.empty:
            st.dataframe(s_df, use_container_width=True, hide_index=True)
        else:
            render_empty_state(title="No Driver Schedules Configured")

        st.markdown("---")
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("##### Assign Driver Shift Schedule")
            with st.form("form_add_schedule", clear_on_submit=True):
                new_driver_id = st.text_input("Driver ID / Name:", placeholder="e.g. DRV-8802").strip()
                sched_vid = st.selectbox(
                    "Assigned Vehicle ID:",
                    options=v_df["vehicle_id"].tolist() if not v_df.empty else [],
                )
                shift_start = st.text_input("Shift Start Time (HH:MM):", value="08:00").strip()
                shift_end = st.text_input("Shift End Time (HH:MM):", value="17:00").strip()
                is_night = st.checkbox("Authorized Night Shift Exemption (22:00–05:00)", value=False)

                submitted_sched = st.form_submit_button("Assign Schedule", type="primary")
                if submitted_sched:
                    success, msg = service.add_driver_schedule(
                        driver_id=new_driver_id,
                        vehicle_id=sched_vid,
                        shift_start=shift_start,
                        shift_end=shift_end,
                        is_night_shift=is_night,
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with sc2:
            st.markdown("##### Remove Shift Schedule")
            if not s_df.empty:
                del_sched_id = st.selectbox(
                    "Select Schedule ID to Remove:",
                    options=s_df["id"].tolist(),
                    format_func=lambda x: f"Schedule #{x} (Driver: {s_df[s_df['id'] == x]['driver_id'].values[0]})",
                )
                if st.button("Delete Schedule", type="secondary"):
                    success, msg = service.delete_driver_schedule(schedule_id=int(del_sched_id))
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # ==========================================================================
    # Tab 3: Incident Review Management
    # ==========================================================================
    with tabs[2]:
        st.subheader("Operational Incident Audit & Formal Resolution")
        st.caption(
            "Review logged incidents and submit supervisor resolution justifications. "
            "Under no circumstances is historical telemetry evidence deleted."
        )

        anom_status = st.radio("Show Incidents:", options=["OPEN", "RESOLVED", "ALL"], horizontal=True)
        anoms = service.get_anomalies(status=anom_status if anom_status != "ALL" else None, limit=200)

        if not anoms.empty:
            st.dataframe(anoms, use_container_width=True, hide_index=True)

            open_anoms = anoms[anoms["status"] != "RESOLVED"]
            if not open_anoms.empty:
                st.markdown("##### Submit Incident Resolution")
                with st.form("form_resolve_incident"):
                    target_id = st.selectbox(
                        "Select Anomaly Incident ID to Resolve:",
                        options=open_anoms["id"].tolist(),
                        format_func=lambda i: f"Incident #{i} - {open_anoms[open_anoms['id'] == i]['incident_type'].values[0]} ({open_anoms[open_anoms['id'] == i]['vehicle_id'].values[0]})",
                    )
                    operator_name = st.text_input("Investigator / Supervisor ID:", value="FleetManager_01")
                    resolution_notes = st.text_area(
                        "Resolution Justification / Operational Finding:",
                        placeholder="Document verified operational reason (e.g. Authorized emergency bypass, road condition variance, or validated sensor calibration error).",
                    )
                    submit_res = st.form_submit_button("Mark Incident as RESOLVED", type="primary")

                    if submit_res:
                        success, msg = service.resolve_anomaly(
                            anomaly_id=int(target_id),
                            resolution_notes=resolution_notes,
                            user_id=operator_name,
                        )
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.success("No open unresolved incidents in this view.")
        else:
            render_empty_state(title="No Incident Records Found")

    # ==========================================================================
    # Tab 4: Administrative Audit Trail
    # ==========================================================================
    with tabs[3]:
        st.subheader("Administrative Mutation Audit Trail")
        st.caption(
            "Immutable record of all administrative configuration actions and supervisor resolutions."
        )

        audit_df = service.get_audit_trail(limit=200)
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            render_empty_state(
                title="No Administrative Audit Events",
                description="Administrative actions performed via this panel will appear here chronologically.",
            )
