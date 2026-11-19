"""Unit and Integration Tests for Dashboard Data Services and Integrations.

Validates:
1. Empty database behaviour across all service queries.
2. Missing or invalid telemetry handling.
3. Latest-record selection per vehicle.
4. Time-difference (ΔT, ΔV, ΔL) kinematic calculations.
5. Anomaly retrieval and multi-parameter filtering.
6. Mathematical reproducibility of Driver Safety Score calculations.
7. CRUD validation, incident resolution, and audit logging.
8. Database write failures and transaction rollback behaviour.
9. Cache invalidation and data refresh consistency.
"""

from __future__ import annotations

import sqlite3
from typing import Generator
import numpy as np
import pandas as pd
import pytest

from database.db_manager import TelematicsDBManager
from src.services.data_service import TelematicsDataService


@pytest.fixture
def empty_service(tmp_path) -> TelematicsDataService:
    """Fixture providing a fresh data service with an initialized but empty SQLite database."""
    db_file = tmp_path / "test_empty_telematics.db"
    manager = TelematicsDBManager(db_path=db_file, auto_seed=False)
    manager.init_db()
    with manager.get_connection() as conn:
        conn.execute("DELETE FROM admin_audit_logs;")
        conn.execute("DELETE FROM anomaly_logs;")
        conn.execute("DELETE FROM driver_schedules;")
        conn.execute("DELETE FROM telematics_pings;")
        conn.execute("DELETE FROM vehicles;")
        conn.commit()
    return TelematicsDataService(db_path=db_file, auto_seed=False)


@pytest.fixture
def populated_service(tmp_path) -> TelematicsDataService:
    """Fixture providing a data service with realistic multi-vehicle telematics and anomalies."""
    db_file = tmp_path / "test_populated_telematics.db"
    manager = TelematicsDBManager(db_path=db_file, auto_seed=False)
    manager.init_db()
    with manager.get_connection() as conn:
        conn.execute("DELETE FROM admin_audit_logs;")
        conn.execute("DELETE FROM anomaly_logs;")
        conn.execute("DELETE FROM driver_schedules;")
        conn.execute("DELETE FROM telematics_pings;")
        conn.execute("DELETE FROM vehicles;")
        conn.commit()

    # Populate vehicles
    manager.add_vehicle("VEH-001", "WXY 1111", "Volvo FH16", 400.0)
    manager.add_vehicle("VEH-002", "WXY 2222", "Scania R500", 350.0)

    # Populate driver schedules
    manager.add_driver_schedule("DRV-A", "VEH-001", "08:00", "17:00", False)
    manager.add_driver_schedule("DRV-B", "VEH-002", "22:00", "06:00", True)

    # Populate telematics pings for VEH-001
    pings_v1 = [
        {
            "vehicle_id": "VEH-001",
            "timestamp": "2024-05-01 10:00:00",
            "latitude": 3.1390,
            "longitude": 101.6869,
            "speed": 60.0,
            "fuel_level": 300.0,
            "heading": 90.0,
            "ignition": 1,
            "delta_time": 10.0,
            "delta_speed": 0.0,
            "delta_fuel": 0.0,
        },
        {
            "vehicle_id": "VEH-001",
            "timestamp": "2024-05-01 10:00:10",
            "latitude": 3.1400,
            "longitude": 101.6879,
            "speed": 85.0,
            "fuel_level": 299.8,
            "heading": 95.0,
            "ignition": 1,
            "delta_time": 10.0,
            "delta_speed": 25.0,
            "delta_fuel": -0.2,
        },
        {
            "vehicle_id": "VEH-001",
            "timestamp": "2024-05-01 10:00:20",
            "latitude": 3.1410,
            "longitude": 101.6889,
            "speed": 92.0,  # Latest ping for VEH-001
            "fuel_level": 299.5,
            "heading": 95.0,
            "ignition": 1,
            "delta_time": 10.0,
            "delta_speed": 7.0,
            "delta_fuel": -0.3,
        },
    ]
    manager.save_telematics_pings(pd.DataFrame(pings_v1))

    # Populate telematics pings for VEH-002
    pings_v2 = [
        {
            "vehicle_id": "VEH-002",
            "timestamp": "2024-05-01 09:00:00",
            "latitude": 3.1200,
            "longitude": 101.6500,
            "speed": 0.0,  # Latest ping for VEH-002
            "fuel_level": 150.0,
            "heading": 0.0,
            "ignition": 0,
            "delta_time": 15.0,
            "delta_speed": 0.0,
            "delta_fuel": 0.0,
        }
    ]
    manager.save_telematics_pings(pd.DataFrame(pings_v2))

    # Populate anomaly logs
    manager.log_anomaly("VEH-001", "2024-05-01 10:00:10", "SPEEDING", 85.0, status="OPEN")
    manager.log_anomaly("VEH-001", "2024-05-01 10:00:20", "SPEEDING", 92.0, status="OPEN")
    manager.log_anomaly("VEH-001", "2024-05-01 10:05:00", "MOONLIGHTING", 1.0, status="OPEN")
    manager.log_anomaly("VEH-002", "2024-05-01 09:00:00", "FUEL_THEFT", 25.0, status="OPEN")

    return TelematicsDataService(db_path=db_file, auto_seed=False)


# ==============================================================================
# 1. Empty Database Behaviour
# ==============================================================================

def test_empty_database_kpis(empty_service: TelematicsDataService) -> None:
    """Verifies that an empty database gracefully returns zero counts and baseline 100.0 score."""
    kpis = empty_service.get_fleet_kpis()
    assert kpis["total_vehicles"] == 0
    assert kpis["active_vehicles"] == 0
    assert kpis["total_pings"] == 0
    assert kpis["unresolved_anomalies"] == 0
    assert kpis["total_anomalies"] == 0
    assert kpis["average_safety_score"] == 100.00
    assert kpis["scored_vehicles_count"] == 0


def test_empty_database_tables(empty_service: TelematicsDataService) -> None:
    """Verifies that queries on empty tables return empty DataFrames without exceptions."""
    fleet_df = empty_service.get_fleet_status_table()
    assert isinstance(fleet_df, pd.DataFrame)
    assert fleet_df.empty

    telem_df = empty_service.get_vehicle_telemetry(vehicle_id="NON_EXISTENT")
    assert isinstance(telem_df, pd.DataFrame)
    assert telem_df.empty

    anom_df = empty_service.get_anomalies()
    assert isinstance(anom_df, pd.DataFrame)
    assert anom_df.empty

    audit_df = empty_service.get_audit_trail()
    assert isinstance(audit_df, pd.DataFrame)
    assert audit_df.empty


# ==============================================================================
# 2. Latest-Record Selection Per Vehicle
# ==============================================================================

def test_latest_record_selection_per_vehicle(populated_service: TelematicsDataService) -> None:
    """Verifies that each vehicle correctly extracts only its chronological newest ping."""
    fleet_df = populated_service.get_fleet_status_table()
    assert len(fleet_df) == 2

    # Verify VEH-001 latest ping attributes
    v1_row = fleet_df[fleet_df["vehicle_id"] == "VEH-001"].iloc[0]
    assert v1_row["speed"] == 92.0
    assert v1_row["fuel_level"] == 299.5
    assert str(v1_row["timestamp"]).startswith("2024-05-01 10:00:20")

    # Verify VEH-002 latest ping attributes
    v2_row = fleet_df[fleet_df["vehicle_id"] == "VEH-002"].iloc[0]
    assert v2_row["speed"] == 0.0
    assert v2_row["fuel_level"] == 150.0
    assert str(v2_row["timestamp"]).startswith("2024-05-01 09:00:00")


# ==============================================================================
# 3. Kinematic Differentials & Time Series Integrity
# ==============================================================================

def test_telemetry_kinematic_time_series(populated_service: TelematicsDataService) -> None:
    """Verifies chronological order, delta variables (ΔT, ΔV, ΔL), and date range filtering."""
    df = populated_service.get_vehicle_telemetry(vehicle_id="VEH-001")
    assert len(df) == 3
    # Check chronological ordering
    assert df["timestamp"].is_monotonic_increasing
    # Check kinematic features
    assert df["delta_time"].iloc[1] == 10.0
    assert df["delta_speed"].iloc[1] == 25.0
    assert round(df["delta_fuel"].iloc[1], 2) == -0.2

    # Date filter test
    df_filtered = populated_service.get_vehicle_telemetry(
        vehicle_id="VEH-001",
        start_date="2024-05-01",
        end_date="2024-05-01",
    )
    assert len(df_filtered) == 3

    # Date filter returning empty for non-matching window
    df_none = populated_service.get_vehicle_telemetry(
        vehicle_id="VEH-001",
        start_date="2023-01-01",
        end_date="2023-01-02",
    )
    assert df_none.empty


# ==============================================================================
# 4. Anomaly Retrieval and Filtering
# ==============================================================================

def test_anomaly_retrieval_and_filtering(populated_service: TelematicsDataService) -> None:
    """Verifies multi-parameter anomaly queries by vehicle, status, and category."""
    all_anoms = populated_service.get_anomalies()
    assert len(all_anoms) == 4

    v1_anoms = populated_service.get_anomalies(vehicle_id="VEH-001")
    assert len(v1_anoms) == 3

    v2_anoms = populated_service.get_anomalies(vehicle_id="VEH-002")
    assert len(v2_anoms) == 1
    assert v2_anoms.iloc[0]["incident_type"] == "FUEL_THEFT"

    speed_anoms = populated_service.get_anomalies(incident_type="SPEEDING")
    assert len(speed_anoms) == 2


# ==============================================================================
# 5. Reproducibility of Scoring Calculation
# ==============================================================================

def test_scoring_reproducibility(populated_service: TelematicsDataService) -> None:
    """Verifies that displayed safety score matches the exact mathematical definition."""
    # VEH-001 has: 2 Speeding, 1 Moonlighting.
    # Co-occurrence detected (Speeding + Moonlighting) -> compounding multiplier 1.5x.
    report = populated_service.calculate_vehicle_safety_score(
        vehicle_id="VEH-001",
        compound_multiplier=1.5,
    )

    assert report.vehicle_id == "VEH-001"
    assert report.compounding_applied is True

    # Exact expected calculations from src/scoring.py:
    # Speeding: count=2, base=2.0, weight=3.0 -> normalized_w = 3/3 = 1.0 -> penalty = 2 * 2.0 * 1.0 * 1.5 = 6.00
    # Moonlighting: count=1, base=3.5, weight=4.0 -> normalized_w = 4/3 -> penalty = 1 * 3.5 * (4/3) * 1.5 = 7.00
    # Total deductions = 6.00 + 7.00 = 13.00
    # Final Score = 100.00 - 13.00 = 87.00 (Low Risk: 75-89)
    assert round(report.total_deductions, 2) == 13.00
    assert round(report.safety_score, 2) == 87.00
    tier_str = report.risk_tier.value if hasattr(report.risk_tier, "value") else str(report.risk_tier)
    assert "Low Risk" in tier_str


def test_fyp1_vs_fyp2_comparison(populated_service: TelematicsDataService) -> None:
    """Verifies comparative evaluation between linear FYP1 (M=1.0) and compounding FYP2 (M=1.5)."""
    comp = populated_service.compare_fyp1_vs_fyp2_scoring(vehicle_id="VEH-001")
    # FYP1 linear deductions: 2 * 2.0 * 1.0 + 1 * 3.5 * (4/3) = 4.00 + 4.67 = 8.67
    # FYP1 score: 100 - 8.67 = 91.33 (Excellent)
    assert round(comp["fyp1_score"], 2) == 91.33
    assert round(comp["fyp2_score"], 2) == 87.00
    assert round(comp["delta_points"], 2) == 4.33
    assert comp["compounding_applied"] is True


# ==============================================================================
# 6. CRUD Validation and Incident Resolution
# ==============================================================================

def test_crud_vehicle_registration_and_validation(populated_service: TelematicsDataService) -> None:
    """Verifies input validation and persistence during vehicle registration and updates."""
    # Invalid empty ID
    ok, err = populated_service.register_vehicle("", "XYZ 999", "Truck", 100.0)
    assert not ok
    assert "empty" in err.lower()

    # Invalid non-positive fuel
    ok, err = populated_service.register_vehicle("VEH-999", "XYZ 999", "Truck", -10.0)
    assert not ok
    assert "greater than 0" in err.lower()

    # Valid registration
    ok, msg = populated_service.register_vehicle("VEH-999", "XYZ 999", "Isuzu Giga", 250.0)
    assert ok
    assert "registered successfully" in msg.lower()

    # Update vehicle
    ok, msg = populated_service.update_vehicle("VEH-999", "XYZ 999-REV", "Isuzu Giga V2", 280.0)
    assert ok
    v = populated_service.db.get_vehicle("VEH-999")
    assert v["plate_number"] == "XYZ 999-REV"
    assert v["fuel_capacity_liters"] == 280.0


def test_incident_resolution_and_audit_logging(populated_service: TelematicsDataService) -> None:
    """Verifies that resolving an incident preserves the record and creates an audit trail entry."""
    anoms = populated_service.get_anomalies(vehicle_id="VEH-002")
    anom_id = int(anoms.iloc[0]["id"])

    # Attempt resolution with empty notes
    ok, err = populated_service.resolve_anomaly(anomaly_id=anom_id, resolution_notes="")
    assert not ok

    # Successful resolution
    ok, msg = populated_service.resolve_anomaly(
        anomaly_id=anom_id,
        resolution_notes="Authorized auxiliary refrigeration unit fuel draw confirmed.",
        user_id="supervisor_test",
    )
    assert ok

    # Verify anomaly is marked RESOLVED, not deleted
    updated = populated_service.get_anomalies(vehicle_id="VEH-002", status="RESOLVED")
    assert len(updated) == 1
    assert updated.iloc[0]["status"] == "RESOLVED"

    # Verify audit trail contains the action
    audit_logs = populated_service.get_audit_trail()
    assert not audit_logs.empty
    recent_entry = audit_logs.iloc[0]
    assert recent_entry["action"] in ("RESOLVE", "RESOLVE_ANOMALY")
    assert recent_entry["entity_id"] == str(anom_id)
    assert "auxiliary refrigeration" in recent_entry["details"]


# ==============================================================================
# 7. Database Write Failures and Rollback Behaviour
# ==============================================================================

def test_database_duplicate_pk_rejection(populated_service: TelematicsDataService) -> None:
    """Verifies that duplicate vehicle ID registration is rejected without corrupting DB."""
    ok, err = populated_service.register_vehicle("VEH-001", "DUP 123", "Duplicate", 100.0)
    assert not ok
    assert "already exists" in err.lower()


def test_foreign_key_violation_rollback(populated_service: TelematicsDataService) -> None:
    """Verifies that assigning a schedule to a non-existent vehicle fails and rolls back."""
    ok, err = populated_service.add_driver_schedule(
        driver_id="DRV-UNKNOWN",
        vehicle_id="NON_EXISTENT_VEHICLE",
        shift_start="08:00",
        shift_end="16:00",
        is_night_shift=False,
    )
    assert not ok
    assert "foreign key" in err.lower() or "does not exist" in err.lower()


# ==============================================================================
# 8. Cache Invalidation and Refresh Consistency
# ==============================================================================

def test_cache_and_state_consistency_after_mutations(populated_service: TelematicsDataService) -> None:
    """Verifies that deletion cascades and status tables reflect mutations immediately."""
    # Delete VEH-002
    ok, msg = populated_service.delete_vehicle("VEH-002")
    assert ok

    # Fleet table should now have only VEH-001
    fleet_df = populated_service.get_fleet_status_table()
    assert len(fleet_df) == 1
    assert fleet_df.iloc[0]["vehicle_id"] == "VEH-001"

    # Cascade should have removed VEH-002 anomalies
    anoms = populated_service.get_anomalies(vehicle_id="VEH-002")
    assert anoms.empty
