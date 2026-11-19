"""Comprehensive Test Suite for TelematicsDBManager SQLite CRUD Engine."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Generator

import pytest

from database.db_manager import DatabaseManagerError, TelematicsDBManager


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Provide a temporary SQLite database path for isolated integration testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_telematics.db"


@pytest.fixture
def db_manager(temp_db_path: Path) -> TelematicsDBManager:
    """Fixture providing an initialized and seeded TelematicsDBManager instance."""
    return TelematicsDBManager(db_path=temp_db_path, auto_init=True, auto_seed=True)


def test_schema_init_and_seeding(db_manager: TelematicsDBManager) -> None:
    """Test schema initialization and mock data seeding for vehicles and driver schedules."""
    # Check seeded vehicles (8_9_8 and 4_2_1)
    vehicles = db_manager.get_all_vehicles()
    assert len(vehicles) >= 2
    v_ids = [v["vehicle_id"] for v in vehicles]
    assert "8_9_8" in v_ids
    assert "4_2_1" in v_ids

    # Vehicle 8_9_8 details
    v_898 = db_manager.get_vehicle("8_9_8")
    assert v_898 is not None
    assert v_898["plate_number"] == "WXY-8988"
    assert v_898["fuel_capacity_liters"] == 450.0

    # Vehicle 4_2_1 details
    v_421 = db_manager.get_vehicle("4_2_1")
    assert v_421 is not None
    assert v_421["plate_number"] == "VBC-4210"
    assert v_421["fuel_capacity_liters"] == 380.0

    # Check shift schedules: 8_9_8 is designated night shift, 4_2_1 is day shift
    assert db_manager.is_night_shift("8_9_8") is True
    assert db_manager.is_night_shift("4_2_1") is False

    # Calling seed_data a second time must be idempotent (no duplicate insertions)
    seeded_again = db_manager.seed_data()
    assert seeded_again is False
    assert len(db_manager.get_all_vehicles()) == len(vehicles)


def test_add_and_get_vehicle(db_manager: TelematicsDBManager) -> None:
    """Test inserting and fetching new commercial vehicles."""
    success = db_manager.add_vehicle(
        vehicle_id="TRUCK_999",
        plate_number="KLM-9999",
        model="Mercedes-Benz Actros 2645",
        fuel_capacity_liters=500.0,
    )
    assert success is True

    vehicle = db_manager.get_vehicle("TRUCK_999")
    assert vehicle is not None
    assert vehicle["vehicle_id"] == "TRUCK_999"
    assert vehicle["plate_number"] == "KLM-9999"
    assert vehicle["model"] == "Mercedes-Benz Actros 2645"
    assert vehicle["fuel_capacity_liters"] == 500.0

    # Duplicate vehicle_id should return False without raising
    duplicate_success = db_manager.add_vehicle(
        vehicle_id="TRUCK_999",
        plate_number="KLM-9999",
        model="Another Model",
        fuel_capacity_liters=400.0,
    )
    assert duplicate_success is False

    # Non-existent vehicle
    assert db_manager.get_vehicle("NON_EXISTENT") is None


def test_driver_schedule_crud(db_manager: TelematicsDBManager) -> None:
    """Test inserting, querying, and checking driver schedules with night-shift flags."""
    # Add a vehicle first
    db_manager.add_vehicle("VAN_101", "B-101", "Ford Transit", 80.0)

    # Insert a night shift schedule
    sched_id = db_manager.add_driver_schedule(
        driver_id="DRV_NIGHT_01",
        vehicle_id="VAN_101",
        shift_start="23:00:00",
        shift_end="07:00:00",
        is_night_shift=True,
    )
    assert sched_id > 0
    assert db_manager.is_night_shift("VAN_101") is True

    schedules = db_manager.get_driver_schedules(vehicle_id="VAN_101")
    assert len(schedules) == 1
    assert schedules[0]["id"] == sched_id
    assert schedules[0]["driver_id"] == "DRV_NIGHT_01"
    assert schedules[0]["is_night_shift"] is True

    # Foreign key constraint check: inserting schedule for non-existent vehicle raises ValueError
    with pytest.raises(ValueError, match="does not exist"):
        db_manager.add_driver_schedule(
            driver_id="DRV_GHOST",
            vehicle_id="GHOST_VEHICLE",
            shift_start="09:00:00",
            shift_end="17:00:00",
            is_night_shift=False,
        )

    # Delete schedule
    deleted = db_manager.delete_driver_schedule(sched_id)
    assert deleted is True
    assert len(db_manager.get_driver_schedules(vehicle_id="VAN_101")) == 0
    assert db_manager.is_night_shift("VAN_101") is False


def test_anomaly_log_crud(db_manager: TelematicsDBManager) -> None:
    """Test logging anomaly incidents, querying, updating status, and deletion."""
    anomaly_id = db_manager.log_anomaly(
        vehicle_id="8_9_8",
        timestamp="2026-03-01T23:45:00Z",
        incident_type="SUDDEN_FUEL_DROP",
        magnitude=-22.5,
        status="OPEN",
    )
    assert anomaly_id > 0

    # Fetch by ID
    log = db_manager.get_anomaly_by_id(anomaly_id)
    assert log is not None
    assert log["vehicle_id"] == "8_9_8"
    assert log["incident_type"] == "SUDDEN_FUEL_DROP"
    assert log["magnitude"] == -22.5
    assert log["status"] == "OPEN"

    # Query all with filters
    open_logs = db_manager.get_anomaly_logs(vehicle_id="8_9_8", status="OPEN")
    assert any(item["id"] == anomaly_id for item in open_logs)

    # Update status to RESOLVED via resolve_anomaly convenience method
    resolve_success = db_manager.resolve_anomaly(anomaly_id)
    assert resolve_success is True

    updated_log = db_manager.get_anomaly_by_id(anomaly_id)
    assert updated_log is not None
    assert updated_log["status"] == "RESOLVED"

    # Filter by resolved status
    resolved_logs = db_manager.get_anomaly_logs(vehicle_id="8_9_8", status="RESOLVED")
    assert any(item["id"] == anomaly_id for item in resolved_logs)

    # Filter by OPEN should no longer include it
    open_logs_after = db_manager.get_anomaly_logs(vehicle_id="8_9_8", status="OPEN")
    assert not any(item["id"] == anomaly_id for item in open_logs_after)

    # Delete anomaly log
    del_success = db_manager.delete_anomaly_log(anomaly_id)
    assert del_success is True
    assert db_manager.get_anomaly_by_id(anomaly_id) is None


def test_delete_vehicle_cascade(db_manager: TelematicsDBManager) -> None:
    """Test that deleting a vehicle cascades to related schedules and anomaly logs."""
    # Add vehicle, schedule, and anomaly
    db_manager.add_vehicle("CASCADE_V1", "CAS-001", "Test Truck", 200.0)
    sched_id = db_manager.add_driver_schedule("DRV_C1", "CASCADE_V1", "08:00", "16:00", False)
    anom_id = db_manager.log_anomaly("CASCADE_V1", "2026-03-01T10:00:00Z", "SPEED_SPIKE", 145.0)

    assert db_manager.get_vehicle("CASCADE_V1") is not None
    assert len(db_manager.get_driver_schedules("CASCADE_V1")) == 1
    assert db_manager.get_anomaly_by_id(anom_id) is not None

    # Delete the vehicle
    deleted = db_manager.delete_vehicle("CASCADE_V1")
    assert deleted is True

    # Vehicle is gone
    assert db_manager.get_vehicle("CASCADE_V1") is None
    # Cascaded schedules and anomaly logs are automatically cleaned up by SQLite foreign keys
    assert len(db_manager.get_driver_schedules("CASCADE_V1")) == 0
    assert db_manager.get_anomaly_by_id(anom_id) is None


def test_default_database_file_creation() -> None:
    """Verify that TelematicsDBManager default initialization creates telematics_core.db."""
    default_mgr = TelematicsDBManager(auto_init=True, auto_seed=True)
    db_file = Path(default_mgr.db_path)
    assert db_file.exists()
    assert db_file.name == "telematics_core.db"

    # Verify tables and mock seed data exist in the default persistent database
    v_898 = default_mgr.get_vehicle("8_9_8")
    assert v_898 is not None
    assert default_mgr.is_night_shift("8_9_8") is True
