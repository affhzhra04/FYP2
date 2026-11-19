"""Comprehensive Test Suite for Telematics Ingestion Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Generator

import pytest

from database.db_manager import TelematicsDBManager
from src.ingest import IngestionPipeline, run_ingestion


@pytest.fixture
def mock_dataset_dir() -> Generator[Path, None, None]:
    """Create a temporary directory structure mimicking the Innonics dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "data"
        sub_dir = base_dir / "G3 New"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # 1. Populated file for vehicle 8_9_8
        records_898 = [
            [
                [
                    {
                        "datetime": "2026-03-01 10:00:00",
                        "latitude": 3.1390,
                        "longitude": 101.6869,
                        "speed": 0.0,
                        "acc": 0,
                        "fuel1_volume": 40.0,
                        "fuel2_volume": 10.0,
                        "heading": 90.0,
                    },
                    {
                        "datetime": "2026-03-01 10:00:10",
                        "latitude": 3.1390,
                        "longitude": 101.6869,
                        "speed": 0.0,
                        "acc": 0,
                        "fuel1_volume": 35.0,  # Drop of 5.0L while stationary
                        "fuel2_volume": 10.0,
                        "heading": 90.0,
                    },
                    {
                        "datetime": "2026-03-01 10:00:20",
                        "latitude": 3.1390,
                        "longitude": 101.6869,
                        "speed": 0.0,
                        "acc": 0,
                        "fuel1_volume": 35.0,  # Persisting drop
                        "fuel2_volume": 10.0,
                        "heading": 90.0,
                    },
                    {
                        "datetime": "2026-03-01 10:00:30",
                        "latitude": 3.1390,
                        "longitude": 101.6869,
                        "speed": 0.0,
                        "acc": 0,
                        "fuel1_volume": 34.9,  # Persisting drop (3rd consecutive reading)
                        "fuel2_volume": 10.0,
                        "heading": 90.0,
                    },
                ]
            ]
        ]
        with open(sub_dir / "8_9_8.json", "w", encoding="utf-8") as f:
            json.dump(records_898, f)

        # 2. Empty nested list file (representing 1_10_1.json)
        empty_payload = [[[]]]
        with open(sub_dir / "1_10_1.json", "w", encoding="utf-8") as f:
            json.dump(empty_payload, f)

        # 3. Completely empty file
        with open(sub_dir / "empty.json", "w", encoding="utf-8") as f:
            f.write("")

        # 4. Malformed JSON file (syntax error)
        with open(sub_dir / "corrupted.json", "w", encoding="utf-8") as f:
            f.write("{ invalid json payload, missing quotes: [")

        yield base_dir


@pytest.fixture
def temp_db() -> Generator[TelematicsDBManager, None, None]:
    """Provide an isolated in-memory TelematicsDBManager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_telematics.db"
        mgr = TelematicsDBManager(db_path=db_path, auto_init=True, auto_seed=True)
        yield mgr


def test_scan_json_files(mock_dataset_dir: Path, temp_db: TelematicsDBManager) -> None:
    """Verify recursive discovery of JSON files."""
    pipeline = IngestionPipeline(data_dir=mock_dataset_dir, db_manager=temp_db)
    found_files = pipeline.scan_json_files()
    assert len(found_files) == 4
    filenames = [p.name for p in found_files]
    assert "8_9_8.json" in filenames
    assert "1_10_1.json" in filenames
    assert "empty.json" in filenames
    assert "corrupted.json" in filenames


def test_parse_single_file_defensive_handling(mock_dataset_dir: Path, temp_db: TelematicsDBManager) -> None:
    """Verify defensive handling of populated, empty, and corrupted JSON files."""
    pipeline = IngestionPipeline(data_dir=mock_dataset_dir, db_manager=temp_db)

    # Populated file
    records, status = pipeline.parse_single_file(mock_dataset_dir / "G3 New" / "8_9_8.json")
    assert status == "POPULATED"
    assert len(records) == 4
    # Check field normalization
    rec0 = records[0]
    assert rec0["vehicle_id"] == "8_9_8"
    assert rec0["fuel_level"] == 50.0  # fuel1 (40) + fuel2 (10)
    assert rec0["ignition"] == 0
    assert rec0["timestamp"] == "2026-03-01 10:00:00"

    # Empty nested list
    empty_recs, empty_status = pipeline.parse_single_file(mock_dataset_dir / "G3 New" / "1_10_1.json")
    assert empty_status == "EMPTY"
    assert len(empty_recs) == 0

    # Completely blank file
    blank_recs, blank_status = pipeline.parse_single_file(mock_dataset_dir / "G3 New" / "empty.json")
    assert blank_status == "EMPTY"
    assert len(blank_recs) == 0

    # Malformed file
    bad_recs, bad_status = pipeline.parse_single_file(mock_dataset_dir / "G3 New" / "corrupted.json")
    assert bad_status == "MALFORMED"
    assert len(bad_recs) == 0


def test_end_to_end_ingestion_pipeline(mock_dataset_dir: Path, temp_db: TelematicsDBManager) -> None:
    """Verify complete ingestion pipeline: reading, cleaning, detecting, and persisting to SQLite."""
    pipeline = IngestionPipeline(data_dir=mock_dataset_dir, db_manager=temp_db)
    report = pipeline.run_pipeline(save_to_db=True)

    assert report["total_files_scanned"] == 4
    assert report["files_populated"] == 1
    assert report["files_empty_or_malformed"] == 3
    assert report["raw_pings_ingested"] == 4
    assert report["valid_clean_pings"] == 4
    assert report["pings_persisted_to_db"] == 4
    assert report["data_retention_rate_pct"] == 100.0

    # Anomalies detected (fuel theft event in mock data)
    assert report["anomalies_detected"] >= 1
    assert report["anomalies_persisted_to_db"] >= 1
    assert "FUEL_THEFT" in report["anomaly_breakdown"]

    # Verify SQLite database storage
    stored_pings = temp_db.get_telematics_pings(vehicle_id="8_9_8")
    assert len(stored_pings) == 4
    assert stored_pings[0]["fuel_level"] == 50.0

    stored_anomalies = temp_db.get_anomaly_logs(vehicle_id="8_9_8")
    assert len(stored_anomalies) >= 1
    assert any(a["incident_type"] == "FUEL_THEFT" for a in stored_anomalies)


def test_ingestion_max_files_limit(mock_dataset_dir: Path, temp_db: TelematicsDBManager) -> None:
    """Verify max_files argument restricts processing appropriately."""
    pipeline = IngestionPipeline(data_dir=mock_dataset_dir, db_manager=temp_db)
    raw_df, metrics = pipeline.ingest_raw_dataset(max_files=1)
    assert metrics["total_files_scanned"] == 1


def test_dry_run_mode_does_not_persist(mock_dataset_dir: Path, temp_db: TelematicsDBManager) -> None:
    """Verify save_to_db=False runs in dry-run mode without database writes."""
    pipeline = IngestionPipeline(data_dir=mock_dataset_dir, db_manager=temp_db)
    report = pipeline.run_pipeline(save_to_db=False)

    assert report["valid_clean_pings"] == 4
    assert report["pings_persisted_to_db"] == 0
    assert report["anomalies_persisted_to_db"] == 0

    # Verify database remains empty of pings
    assert len(temp_db.get_telematics_pings()) == 0
