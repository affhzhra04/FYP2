"""Comprehensive Test Suite for TelematicsCleaner Preprocessing Engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import TelematicsCleaner


@pytest.fixture
def cleaner() -> TelematicsCleaner:
    """Fixture providing a standard TelematicsCleaner instance."""
    return TelematicsCleaner()


@pytest.fixture
def sample_telematics_data() -> pd.DataFrame:
    """Fixture providing a multi-vehicle raw telematics dataset with known edge cases."""
    return pd.DataFrame(
        {
            "vehicle_id": [
                "TRUCK_01",
                "TRUCK_01",
                "TRUCK_01",
                "TRUCK_01",  # Duplicate ping with row 2
                "TRUCK_01",
                "TRUCK_02",
                "TRUCK_02",
                "TRUCK_03",  # Invalid latitude
                "TRUCK_03",  # Invalid speed (> 160)
                "TRUCK_03",  # Invalid fuel (-999 sentinel)
            ],
            "timestamp": [
                "2026-03-01 08:00:00",
                "2026-03-01 08:00:10",
                "2026-03-01 08:00:20",
                "2026-03-01 08:00:20",  # Exact duplicate timestamp for TRUCK_01
                "2026-03-01 08:00:35",
                "2026-03-01 09:00:00",
                "2026-03-01 09:00:15",
                "2026-03-01 10:00:00",
                "2026-03-01 10:00:10",
                "2026-03-01 10:00:20",
            ],
            "latitude": [
                37.7749,
                37.7750,
                37.7752,
                37.7752,
                37.7755,
                40.7128,
                40.7130,
                95.0000,  # Invalid: > 90
                40.7128,
                40.7128,
            ],
            "longitude": [
                -122.4194,
                -122.4190,
                -122.4185,
                -122.4185,
                -122.4180,
                -74.0060,
                -74.0055,
                -74.0060,
                -74.0060,
                -74.0060,
            ],
            "speed": [
                0.0,
                45.5,
                62.0,
                62.0,
                58.0,
                80.0,
                85.0,
                50.0,
                185.0,  # Invalid: > 160
                50.0,
            ],
            "fuel_level": [
                90.0,
                89.8,
                89.5,
                89.5,
                89.1,
                120.0,
                119.5,
                50.0,
                50.0,
                -999.0,  # Invalid: sentinel value
            ],
        }
    )


def test_coordinate_validity(cleaner: TelematicsCleaner) -> None:
    """Test filtering of latitude [-90, 90], longitude [-180, 180], and NaNs."""
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1", "V1", "V1", "V1", "V1", "V1"],
            "timestamp": [f"2026-01-01 10:00:0{i}" for i in range(6)],
            "latitude": [0.0, -90.0, 90.0, -90.001, 90.001, np.nan],
            "longitude": [0.0, -180.0, 180.0, 0.0, 0.0, 10.0],
            "speed": [50.0] * 6,
            "fuel_level": [50.0] * 6,
        }
    )
    cleaned_df, audit = cleaner.clean_data(df)

    assert audit["initial_records"] == 6
    assert audit["dropped_invalid_coordinates"] == 3
    assert audit["valid_records"] == 3
    assert len(cleaned_df) == 3

    # Longitude out-of-range test
    df_lon = pd.DataFrame(
        {
            "vehicle_id": ["V1", "V1", "V1"],
            "timestamp": ["2026-01-01 10:00:00", "2026-01-01 10:00:10", "2026-01-01 10:00:20"],
            "latitude": [10.0, 10.0, 10.0],
            "longitude": [-180.001, 180.001, np.nan],
            "speed": [50.0, 50.0, 50.0],
            "fuel_level": [50.0, 50.0, 50.0],
        }
    )
    cleaned_lon, audit_lon = cleaner.clean_data(df_lon)
    assert audit_lon["dropped_invalid_coordinates"] == 3
    assert audit_lon["valid_records"] == 0


def test_speed_sanitization(cleaner: TelematicsCleaner) -> None:
    """Test speed boundaries [0, 160] km/h and NaN rejection."""
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1"] * 5,
            "timestamp": [f"2026-01-01 10:00:0{i}" for i in range(5)],
            "latitude": [10.0] * 5,
            "longitude": [10.0] * 5,
            "speed": [-0.1, 0.0, 160.0, 160.1, np.nan],
            "fuel_level": [50.0] * 5,
        }
    )
    cleaned_df, audit = cleaner.clean_data(df)

    assert audit["initial_records"] == 5
    assert audit["dropped_invalid_speed"] == 3
    assert audit["valid_records"] == 2
    assert list(cleaned_df["speed"]) == [0.0, 160.0]


def test_fuel_reading_integrity(cleaner: TelematicsCleaner) -> None:
    """Test fuel integrity checks rejecting values < 0 and sentinels (-999, -9999)."""
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1"] * 6,
            "timestamp": [f"2026-01-01 10:00:0{i}" for i in range(6)],
            "latitude": [10.0] * 6,
            "longitude": [10.0] * 6,
            "speed": [50.0] * 6,
            "fuel_level": [-999.0, -9999.0, -1.0, -0.01, 0.0, 45.0],
        }
    )
    cleaned_df, audit = cleaner.clean_data(df)

    assert audit["initial_records"] == 6
    assert audit["dropped_invalid_fuel"] == 4
    assert audit["valid_records"] == 2
    assert list(cleaned_df["fuel_level"]) == [0.0, 45.0]


def test_timestamp_parsing_and_deduplication(cleaner: TelematicsCleaner) -> None:
    """Test timestamp conversion to standard UTC datetime64[ns], sorting, and deduplication."""
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1", "V1", "V2", "V1"],
            "timestamp": [
                "2026-01-01 10:00:10Z",
                "2026-01-01 10:00:00Z",  # Needs sorting before 10:00:10
                "2026-01-01 09:00:00Z",
                "2026-01-01 10:00:10Z",  # Duplicate timestamp for V1
            ],
            "latitude": [10.0, 10.0, 20.0, 10.0],
            "longitude": [10.0, 10.0, 20.0, 10.0],
            "speed": [50.0, 40.0, 60.0, 50.0],
            "fuel_level": [50.0, 50.5, 70.0, 50.0],
        }
    )
    cleaned_df, audit = cleaner.clean_data(df)

    assert audit["dropped_duplicate_timestamps"] == 1
    assert audit["valid_records"] == 3

    # Check standard UTC datetime64[ns] dtype
    assert cleaned_df["timestamp"].dtype == "datetime64[ns]"

    # Verify sorting order: V1 first (sorted by timestamp), then V2
    v1_timestamps = list(cleaned_df[cleaned_df["vehicle_id"] == "V1"]["timestamp"])
    assert v1_timestamps == [
        pd.Timestamp("2026-01-01 10:00:00"),
        pd.Timestamp("2026-01-01 10:00:10"),
    ]


def test_feature_engineering_precomputations(cleaner: TelematicsCleaner) -> None:
    """Test vectorized precomputations of delta_t, delta_v, and delta_l grouped by vehicle_id."""
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1", "V1", "V2", "V2"],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:00:15",  # dt = 15s, dv = +20, dl = -1.5
                "2026-01-01 12:00:00",  # New vehicle V2 (must have NaN deltas)
                "2026-01-01 12:00:30",  # dt = 30s, dv = -10, dl = -2.0
            ],
            "latitude": [10.0, 10.1, 20.0, 20.1],
            "longitude": [10.0, 10.1, 20.0, 20.1],
            "speed": [40.0, 60.0, 80.0, 70.0],
            "fuel_level": [50.0, 48.5, 100.0, 98.0],
        }
    )
    cleaned_df, audit = cleaner.clean_data(df)

    assert "delta_time" in cleaned_df.columns
    assert "delta_speed" in cleaned_df.columns
    assert "delta_fuel" in cleaned_df.columns
    assert "delta_t" in cleaned_df.columns
    assert "delta_v" in cleaned_df.columns
    assert "delta_l" in cleaned_df.columns

    # First row of V1
    row0 = cleaned_df.iloc[0]
    assert row0["vehicle_id"] == "V1"
    assert np.isnan(row0["delta_time"])
    assert np.isnan(row0["delta_speed"])
    assert np.isnan(row0["delta_fuel"])

    # Second row of V1
    row1 = cleaned_df.iloc[1]
    assert row1["vehicle_id"] == "V1"
    assert row1["delta_time"] == 15.0
    assert row1["delta_speed"] == 20.0
    assert pytest.approx(row1["delta_fuel"]) == -1.5

    # First row of V2 (boundary check: must NOT compute difference from V1)
    row2 = cleaned_df.iloc[2]
    assert row2["vehicle_id"] == "V2"
    assert np.isnan(row2["delta_time"])
    assert np.isnan(row2["delta_speed"])
    assert np.isnan(row2["delta_fuel"])

    # Second row of V2
    row3 = cleaned_df.iloc[3]
    assert row3["vehicle_id"] == "V2"
    assert row3["delta_time"] == 30.0
    assert row3["delta_speed"] == -10.0
    assert pytest.approx(row3["delta_fuel"]) == -2.0


def test_complete_audit_report(cleaner: TelematicsCleaner, sample_telematics_data: pd.DataFrame) -> None:
    """Test that all exact keys and values are properly formatted in the audit report."""
    cleaned_df, audit = cleaner.clean_data(sample_telematics_data)

    required_keys = {
        "initial_records",
        "valid_records",
        "dropped_invalid_coordinates",
        "dropped_invalid_speed",
        "dropped_invalid_fuel",
        "dropped_duplicate_timestamps",
        "data_retention_rate_pct",
    }
    assert required_keys.issubset(audit.keys())

    assert audit["initial_records"] == 10
    assert audit["dropped_invalid_coordinates"] == 1  # lat = 95.0
    assert audit["dropped_invalid_speed"] == 1        # speed = 185.0
    assert audit["dropped_invalid_fuel"] == 1         # fuel = -999.0
    assert audit["dropped_duplicate_timestamps"] == 1 # duplicate at 08:00:20
    assert audit["valid_records"] == 6

    # Retention rate: 6 / 10 * 100 = 60.0%
    assert audit["data_retention_rate_pct"] == 60.0
    assert len(cleaned_df) == 6


def test_json_records_ingestion(cleaner: TelematicsCleaner) -> None:
    """Test ingestion and cleaning of JSON records (List[Dict])."""
    records = [
        {
            "vehicle_id": "VAN_01",
            "timestamp": "2026-03-01 10:00:00",
            "latitude": 3.1390,
            "longitude": 101.6869,
            "speed": 50.0,
            "fuel_level": 40.0,
        },
        {
            "vehicle_id": "VAN_01",
            "timestamp": "2026-03-01 10:00:10",
            "latitude": 3.1392,
            "longitude": 101.6872,
            "speed": 55.0,
            "fuel_level": 39.8,
        },
    ]

    cleaned_df, audit = cleaner.clean_data(records)
    assert isinstance(cleaned_df, pd.DataFrame)
    assert audit["initial_records"] == 2
    assert audit["valid_records"] == 2
    assert audit["data_retention_rate_pct"] == 100.0
    assert len(cleaned_df) == 2


def test_column_alias_resolution(cleaner: TelematicsCleaner) -> None:
    """Test plug-and-play resolution of common IoT column aliases."""
    df_aliases = pd.DataFrame(
        {
            "vid": ["V1", "V1"],
            "time": ["2026-01-01 10:00:00", "2026-01-01 10:00:10"],
            "lat": [10.0, 10.1],
            "lon": [20.0, 20.1],
            "velocity": [30.0, 35.0],
            "fuel": [50.0, 49.5],
        }
    )
    cleaned_df, audit = cleaner.clean_data(df_aliases)
    assert audit["valid_records"] == 2
    assert "delta_time" in cleaned_df.columns
    assert "delta_speed" in cleaned_df.columns
    assert "delta_fuel" in cleaned_df.columns


def test_empty_dataframe_handling(cleaner: TelematicsCleaner) -> None:
    """Test graceful handling of empty datasets."""
    empty_df = pd.DataFrame()
    cleaned_df, audit = cleaner.clean_data(empty_df)

    assert audit["initial_records"] == 0
    assert audit["valid_records"] == 0
    assert audit["data_retention_rate_pct"] == 0.0
    assert len(cleaned_df) == 0


def test_performance_over_300k_rows(cleaner: TelematicsCleaner) -> None:
    """Stress test verifying sub-second, highly-optimized performance over 300,000+ rows."""
    n_records = 320_000
    n_vehicles = 500

    np.random.seed(42)
    vehicle_pool = [f"FLEET_VEHICLE_{i:04d}" for i in range(n_vehicles)]

    # Generate synthetic telematics data with known anomalies
    v_ids = np.random.choice(vehicle_pool, size=n_records)
    latitudes = np.random.uniform(30.0, 45.0, size=n_records)
    longitudes = np.random.uniform(-120.0, -70.0, size=n_records)
    speeds = np.random.uniform(0.0, 130.0, size=n_records)
    fuel_levels = np.random.uniform(10.0, 100.0, size=n_records)

    # Inject anomalies
    latitudes[0:1000] = 95.0  # Invalid coordinates
    speeds[1000:2000] = 180.0  # Invalid speeds
    fuel_levels[2000:3000] = -999.0  # Sentinel fuel

    # Random timestamps across 2 days using 64-bit epoch seconds
    epoch_seconds = np.random.randint(1_772_323_200, 1_772_496_000, size=n_records, dtype=np.int64)
    timestamps = pd.to_datetime(epoch_seconds, unit="s")

    df_large = pd.DataFrame(
        {
            "vehicle_id": v_ids,
            "timestamp": timestamps,
            "latitude": latitudes,
            "longitude": longitudes,
            "speed": speeds,
            "fuel_level": fuel_levels,
        }
    )

    start_time = time.perf_counter()
    cleaned_df, audit = cleaner.clean_data(df_large)
    elapsed_time = time.perf_counter() - start_time

    # Performance assertion: 320,000 rows processed in well under 3.0 seconds
    assert elapsed_time < 3.0, f"Processing {n_records} rows took {elapsed_time:.3f}s (exceeded 3.0s)"
    assert audit["initial_records"] == n_records
    assert audit["valid_records"] > 0
    assert audit["dropped_invalid_coordinates"] >= 1000
    assert audit["dropped_invalid_speed"] >= 1000
    assert audit["dropped_invalid_fuel"] >= 1000
    assert 0.0 < audit["data_retention_rate_pct"] <= 100.0
