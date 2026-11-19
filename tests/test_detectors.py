"""Comprehensive Unit and Integration Test Suite for Telematics Anomaly Detectors."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from database.db_manager import TelematicsDBManager
from src.detectors import (
    EnhancedFuelDetector,
    EnhancedMoonlightDetector,
    KinematicAggressionDetector,
)


@pytest.fixture
def temp_db_manager() -> Generator[TelematicsDBManager, None, None]:
    """Provide an isolated, seeded TelematicsDBManager instance for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_telematics.db"
        mgr = TelematicsDBManager(db_path=db_path, auto_init=True, auto_seed=True)
        yield mgr


# ==============================================================================
# EnhancedFuelDetector Tests
# ==============================================================================


def test_fuel_theft_stationary_persistent_drop() -> None:
    """Verify fuel theft detection when vehicle is stationary and drop persists for >= 3 readings."""
    detector = EnhancedFuelDetector(
        theft_threshold_liters=0.3,
        min_consecutive_readings=3,
        speed_tolerance_kmh=0.0,
    )

    # 5 stationary readings with a sudden sustained drop of 4.0L
    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01"] * 5,
            "timestamp": [f"2026-03-01 10:00:0{i}" for i in range(5)],
            "speed": [0.0, 0.0, 0.0, 0.0, 0.0],
            "ignition": ["OFF", "OFF", "OFF", "OFF", "OFF"],
            "fuel_level": [50.0, 46.0, 45.9, 45.8, 45.8],  # Drop of 4.0L persisting for 4 readings
        }
    )

    enriched_df, anomalies = detector.detect(df)

    # Anomaly must be triggered on the 3rd consecutive reading of the drop
    assert len(anomalies) == 1
    assert anomalies[0].incident_type == "FUEL_THEFT"
    assert anomalies[0].vehicle_id == "V_01"
    assert pytest.approx(anomalies[0].magnitude, abs=0.1) == -4.2  # drop from 50.0 to 45.8

    # Ensure DataFrame columns are populated
    assert enriched_df["is_fuel_theft"].sum() >= 1
    assert "FUEL_THEFT" in enriched_df["fuel_event"].values


def test_fuel_sloshing_rebound_rejection() -> None:
    """Verify ultrasonic acoustic sloshing rebound does NOT trigger a false fuel theft alert."""
    detector = EnhancedFuelDetector(
        theft_threshold_liters=0.3,
        min_consecutive_readings=3,
    )

    # Temporary dip of 0.8L that rebounds after 1 reading
    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01"] * 5,
            "timestamp": [f"2026-03-01 10:00:0{i}" for i in range(5)],
            "speed": [0.0] * 5,
            "ignition": ["OFF"] * 5,
            "fuel_level": [50.0, 49.2, 50.0, 49.9, 50.0],  # Dips to 49.2 then rebounds to 50.0
        }
    )

    enriched_df, anomalies = detector.detect(df)

    # Slosh rebound should prevent false alerts
    theft_anomalies = [a for a in anomalies if a.incident_type == "FUEL_THEFT"]
    assert len(theft_anomalies) == 0
    assert not enriched_df["is_fuel_theft"].any()


def test_refuel_event_classification() -> None:
    """Verify that a positive fuel jump > 5.0L is classified as REFUEL and resets baseline."""
    detector = EnhancedFuelDetector(refuel_threshold_liters=5.0)

    # Refuel surge from 20L to 80L
    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01"] * 4,
            "timestamp": [f"2026-03-01 10:00:0{i}" for i in range(4)],
            "speed": [0.0] * 4,
            "ignition": ["OFF"] * 4,
            "fuel_level": [20.0, 20.0, 80.0, 79.9],  # +60.0L refuel jump
        }
    )

    enriched_df, anomalies = detector.detect(df)

    refuels = [a for a in anomalies if a.incident_type == "REFUEL"]
    assert len(refuels) == 1
    assert refuels[0].magnitude == 60.0
    assert enriched_df.iloc[2]["fuel_event"] == "REFUEL"
    # No fuel theft should be triggered
    assert not enriched_df["is_fuel_theft"].any()


def test_fuel_drop_while_driving_not_theft() -> None:
    """Verify that normal engine consumption while vehicle is moving is NOT flagged as theft."""
    detector = EnhancedFuelDetector(theft_threshold_liters=0.3)

    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01"] * 4,
            "timestamp": [f"2026-03-01 10:00:0{i}" for i in range(4)],
            "speed": [65.0, 70.0, 68.0, 72.0],  # Vehicle moving on highway
            "ignition": ["ON", "ON", "ON", "ON"],
            "fuel_level": [50.0, 49.5, 49.0, 48.5],  # Fuel drops steadily
        }
    )

    enriched_df, anomalies = detector.detect(df)
    assert len(anomalies) == 0
    assert not enriched_df["is_fuel_theft"].any()


# ==============================================================================
# EnhancedMoonlightDetector Tests
# ==============================================================================


def test_moonlight_night_shift_authorized_vs_unauthorized(temp_db_manager: TelematicsDBManager) -> None:
    """Verify Ghost Hours (22:00–04:59) penalty exemption for authorized night shift vs unauthorized."""
    detector = EnhancedMoonlightDetector(db_manager=temp_db_manager)

    # Seeded: vehicle '8_9_8' has authorized night shift; '4_2_1' is DAY shift only
    df = pd.DataFrame(
        {
            "vehicle_id": ["8_9_8", "4_2_1", "4_2_1"],
            "timestamp": [
                "2026-03-01 23:30:00",  # Ghost hour (23:30), but 8_9_8 is AUTHORIZED
                "2026-03-01 23:30:00",  # Ghost hour (23:30), 4_2_1 is UNAUTHORIZED -> VIOLATION
                "2026-03-01 11:00:00",  # Daytime (11:00), 4_2_1 is AUTHORIZED DAY DRIVING
            ],
            "latitude": [3.1390, 3.1390, 3.1390],  # At base (no geofence breach)
            "longitude": [101.6869, 101.6869, 101.6869],
            "speed": [45.0, 50.0, 50.0],
        }
    )

    enriched_df, anomalies = detector.detect(df)

    # Row 0: 8_9_8 at 23:30 is night-shift authorized
    assert bool(enriched_df.iloc[0]["is_night_shift_authorized"]) is True
    assert bool(enriched_df.iloc[0]["is_unauthorized_night_driving"]) is False

    # Row 1: 4_2_1 at 23:30 is unauthorized night driving
    assert bool(enriched_df.iloc[1]["is_night_shift_authorized"]) is False
    assert bool(enriched_df.iloc[1]["is_unauthorized_night_driving"]) is True

    # Row 2: 4_2_1 at 11:00 is daytime
    assert bool(enriched_df.iloc[2]["is_ghost_hours"]) is False
    assert bool(enriched_df.iloc[2]["is_unauthorized_night_driving"]) is False

    ghost_anomalies = [a for a in anomalies if a.incident_type == "UNAUTHORIZED_NIGHT_DRIVING"]
    assert len(ghost_anomalies) == 1
    assert ghost_anomalies[0].vehicle_id == "4_2_1"


def test_moonlight_geofence_breach(temp_db_manager: TelematicsDBManager) -> None:
    """Verify Haversine great-circle distance calculation and geofence breach alert."""
    detector = EnhancedMoonlightDetector(
        db_manager=temp_db_manager,
        base_latitude=3.1390,
        base_longitude=101.6869,
        max_distance_km=50.0,
    )

    # Point 1: near base (~0 km); Point 2: Penang/far point (~300 km)
    df = pd.DataFrame(
        {
            "vehicle_id": ["8_9_8", "8_9_8"],
            "timestamp": ["2026-03-01 10:00:00", "2026-03-01 14:00:00"],
            "latitude": [3.1400, 5.4164],       # Point 2 is Penang (~300 km from KL base)
            "longitude": [101.6900, 100.3327],
            "speed": [40.0, 80.0],
        }
    )

    enriched_df, anomalies = detector.detect(df)

    # Point 1 within 50 km
    assert enriched_df.iloc[0]["distance_from_base_km"] < 5.0
    assert bool(enriched_df.iloc[0]["is_geofence_breach"]) is False

    # Point 2 outside 50 km
    assert enriched_df.iloc[1]["distance_from_base_km"] > 250.0
    assert bool(enriched_df.iloc[1]["is_geofence_breach"]) is True

    geofence_anomalies = [a for a in anomalies if a.incident_type == "GEOFENCE_BREACH"]
    assert len(geofence_anomalies) == 1
    assert geofence_anomalies[0].vehicle_id == "8_9_8"


# ==============================================================================
# KinematicAggressionDetector Tests
# ==============================================================================


def test_kinematic_harsh_braking_and_accel() -> None:
    """Verify longitudinal acceleration (a = ΔV / ΔT) for harsh braking and harsh acceleration."""
    detector = KinematicAggressionDetector(
        harsh_braking_threshold_mps2=-3.0,
        harsh_accel_threshold_mps2=2.5,
    )

    # Row 1 -> Row 2: decel from 80 km/h to 30 km/h in 3s
    # ΔV = -50 km/h = -13.889 m/s, a = -13.889 / 3 = -4.63 m/s² <= -3.0 (HARSH BRAKING)
    # Row 2 -> Row 3: accel from 30 km/h to 70 km/h in 3s
    # ΔV = +40 km/h = +11.111 m/s, a = +11.111 / 3 = +3.70 m/s² >= 2.5 (HARSH ACCELERATION)
    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01", "V_01", "V_01"],
            "timestamp": [
                "2026-03-01 10:00:00",
                "2026-03-01 10:00:03",
                "2026-03-01 10:00:06",
            ],
            "speed": [80.0, 30.0, 70.0],
            "delta_time": [np.nan, 3.0, 3.0],
            "delta_speed": [np.nan, -50.0, 40.0],
        }
    )

    enriched_df, anomalies = detector.detect(df)

    assert bool(enriched_df.iloc[1]["is_harsh_braking"]) is True
    assert bool(enriched_df.iloc[1]["is_harsh_accel"]) is False
    assert enriched_df.iloc[1]["longitudinal_accel_mps2"] < -4.0

    assert bool(enriched_df.iloc[2]["is_harsh_braking"]) is False
    assert bool(enriched_df.iloc[2]["is_harsh_accel"]) is True
    assert enriched_df.iloc[2]["longitudinal_accel_mps2"] > 3.0

    types = [a.incident_type for a in anomalies]
    assert "HARSH_BRAKING" in types
    assert "HARSH_ACCELERATION" in types


def test_kinematic_speeding() -> None:
    """Verify speeding detection (> 90 km/h)."""
    detector = KinematicAggressionDetector(speed_limit_kmh=90.0)

    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01", "V_01"],
            "timestamp": ["2026-03-01 10:00:00", "2026-03-01 10:00:10"],
            "speed": [85.0, 115.0],
            "delta_time": [np.nan, 10.0],
            "delta_speed": [np.nan, 30.0],
        }
    )

    enriched_df, anomalies = detector.detect(df)
    assert not bool(enriched_df.iloc[0]["is_speeding"])
    assert bool(enriched_df.iloc[1]["is_speeding"])

    speeding_events = [a for a in anomalies if a.incident_type == "OVERSPEEDING"]
    assert len(speeding_events) == 1
    assert speeding_events[0].magnitude == 115.0


def test_kinematic_harsh_cornering_with_modular_heading() -> None:
    """Verify lateral angular velocity (omega = Δheading / ΔT) across 360° boundary."""
    detector = KinematicAggressionDetector(
        harsh_cornering_threshold_dps=25.0,
        min_cornering_speed_kmh=15.0,
    )

    # Heading turn from 355° to 15° (+20° change) in 0.5s -> omega = 40 deg/s >= 25 deg/s
    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01", "V_01"],
            "timestamp": ["2026-03-01 10:00:00.000", "2026-03-01 10:00:00.500"],
            "speed": [35.0, 35.0],
            "heading": [355.0, 15.0],
            "delta_time": [np.nan, 0.5],
            "delta_speed": [np.nan, 0.0],
        }
    )

    enriched_df, anomalies = detector.detect(df)

    assert bool(enriched_df.iloc[1]["is_harsh_cornering"]) is True
    assert enriched_df.iloc[1]["angular_velocity_dps"] == 40.0

    cornering_events = [a for a in anomalies if a.incident_type == "HARSH_CORNERING"]
    assert len(cornering_events) == 1


def test_kinematic_excessive_idling() -> None:
    """Verify excessive idling flag when stationary time exceeds 300 seconds."""
    detector = KinematicAggressionDetector(excessive_idling_threshold_sec=300.0)

    # Stationary sequence totaling 360s
    df = pd.DataFrame(
        {
            "vehicle_id": ["V_01"] * 4,
            "timestamp": [
                "2026-03-01 10:00:00",
                "2026-03-01 10:02:00",  # +120s
                "2026-03-01 10:04:00",  # +120s (cumulative 240s)
                "2026-03-01 10:06:00",  # +120s (cumulative 360s > 300s -> EXCESSIVE IDLE)
            ],
            "speed": [0.0, 0.0, 0.0, 0.0],
            "delta_time": [np.nan, 120.0, 120.0, 120.0],
            "delta_speed": [np.nan, 0.0, 0.0, 0.0],
        }
    )

    enriched_df, anomalies = detector.detect(df)

    assert bool(enriched_df.iloc[1]["is_excessive_idling"]) is False
    assert bool(enriched_df.iloc[2]["is_excessive_idling"]) is False
    assert bool(enriched_df.iloc[3]["is_excessive_idling"]) is True

    idling_events = [a for a in anomalies if a.incident_type == "EXCESSIVE_IDLING"]
    assert len(idling_events) == 1
