"""Production-grade Telematics Fleet Data Ingestion and Pipeline Orchestrator.

This module automates the ingestion, normalization, quality cleaning, feature precomputation,
anomaly detection, and database persistence for the Innonics commercial IoT telematics dataset.

Pipeline Workflow:
  1. Scans the local data/ directory for all JSON files with robust error handling.
  2. Parses and flattens nested JSON payloads into a consolidated Pandas DataFrame.
  3. Executes TelematicsCleaner (coordinate validation, speed clamping, deduplication, ΔT, ΔV, ΔL).
  4. Runs multi-condition anomaly detectors (EnhancedFuel, EnhancedMoonlight, KinematicAggression).
  5. Persists cleaned pings and detected incidents into database/telematics_core.db.
  6. Outputs a formatted terminal audit summary report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# Ensure project root is present in sys.path for direct CLI script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DEFAULT_DB_PATH, TelematicsDBManager
from src.detectors import (
    AnomalyEvent,
    EnhancedFuelDetector,
    EnhancedMoonlightDetector,
    KinematicAggressionDetector,
)
from src.preprocessing import TelematicsCleaner

# Configure module-level logger
logger = logging.getLogger("src.ingest")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [ingest]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default location for raw datasets
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class IngestionPipeline:
    """High-throughput Innonics fleet telematics ingestion and persistence engine.

    Attributes:
        data_dir (Path): Base directory containing raw JSON files.
        db_manager (TelematicsDBManager): Target SQLite database manager.
        cleaner (TelematicsCleaner): Data quality and feature precomputing engine.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        db_manager: Optional[TelematicsDBManager] = None,
    ) -> None:
        """Initialize the IngestionPipeline with dataset directory and database connection."""
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR).resolve()
        if db_manager is not None:
            self.db_manager = db_manager
        else:
            self.db_manager = TelematicsDBManager(db_path=db_path or DEFAULT_DB_PATH)

        self.cleaner = TelematicsCleaner()
        self.fuel_detector = EnhancedFuelDetector()
        self.moonlight_detector = EnhancedMoonlightDetector(db_manager=self.db_manager)
        self.kinematic_detector = KinematicAggressionDetector()

    def scan_json_files(self) -> List[Path]:
        """Recursively scan data directory for all JSON files.

        Returns
        -------
        List[Path]
            Sorted list of discovered JSON file paths.
        """
        if not self.data_dir.exists():
            logger.warning("Data directory does not exist: %s", self.data_dir)
            return []

        # Recursively search for all .json files
        json_files = sorted(list(self.data_dir.rglob("*.json")))
        logger.info("Discovered %d JSON files in %s", len(json_files), self.data_dir)
        return json_files

    @staticmethod
    def _extract_nested_records(payload: Any) -> List[Dict[str, Any]]:
        """Recursively flatten arbitrarily nested lists into a list of record dictionaries."""
        if isinstance(payload, list):
            flattened = []
            for element in payload:
                flattened.extend(IngestionPipeline._extract_nested_records(element))
            return flattened
        if isinstance(payload, dict):
            return [payload]
        return []

    def parse_single_file(self, file_path: Path) -> Tuple[List[Dict[str, Any]], str]:
        """Parse a single JSON file with defensive error handling for malformed content.

        Parameters
        ----------
        file_path : Path
            Path to the JSON file to parse.

        Returns
        -------
        Tuple[List[Dict[str, Any]], str]
            - Extracted and normalized records.
            - Status code: 'POPULATED', 'EMPTY', or 'MALFORMED'.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return [], "EMPTY"
                data = json.loads(content)

            raw_records = self._extract_nested_records(data)
            if not raw_records:
                return [], "EMPTY"

            stem_vehicle_id = file_path.stem
            normalized: List[Dict[str, Any]] = []

            for item in raw_records:
                if not isinstance(item, dict):
                    continue

                rec = dict(item)

                # Ensure vehicle_id is present (fallback to filename stem e.g. '8_9_8')
                if "vehicle_id" not in rec or not rec["vehicle_id"]:
                    rec["vehicle_id"] = stem_vehicle_id

                # Resolve fuel level from dual-tank sensor columns if present
                if "fuel_level" not in rec and "fuel1_volume" in rec:
                    f1 = float(rec.get("fuel1_volume") or 0.0)
                    f2 = float(rec.get("fuel2_volume") or 0.0)
                    rec["fuel_level"] = f1 + f2

                # Resolve ignition status from accessory column (acc) if present
                if "ignition" not in rec and "acc" in rec:
                    rec["ignition"] = rec["acc"]

                # Resolve timestamp from datetime column if present
                if "timestamp" not in rec and "datetime" in rec:
                    rec["timestamp"] = rec["datetime"]

                normalized.append(rec)

            return normalized, "POPULATED"

        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            logger.debug("Malformed JSON skipped in %s: %s", file_path.name, exc)
            return [], "MALFORMED"
        except OSError as exc:
            logger.warning("I/O error reading %s: %s", file_path.name, exc)
            return [], "MALFORMED"

    def ingest_raw_dataset(
        self,
        max_files: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Read and parse all discovered JSON files into a consolidated DataFrame.

        Parameters
        ----------
        max_files : Optional[int], optional
            Optional limit on number of files to process (useful for testing).

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, int]]
            - Consolidated raw DataFrame.
            - File parsing metrics dictionary.
        """
        all_files = self.scan_json_files()
        if max_files is not None:
            all_files = all_files[:max_files]

        total_files = len(all_files)
        file_metrics = {
            "total_files_scanned": total_files,
            "files_populated": 0,
            "files_empty": 0,
            "files_malformed": 0,
            "raw_pings_ingested": 0,
        }

        all_records: List[Dict[str, Any]] = []

        for idx, file_path in enumerate(all_files, start=1):
            records, status = self.parse_single_file(file_path)
            if status == "POPULATED":
                file_metrics["files_populated"] += 1
                all_records.extend(records)
            elif status == "EMPTY":
                file_metrics["files_empty"] += 1
            else:
                file_metrics["files_malformed"] += 1

            if idx % 100 == 0 or idx == total_files:
                logger.info("Processed %d / %d files (%d records accumulated)...", idx, total_files, len(all_records))

        file_metrics["raw_pings_ingested"] = len(all_records)
        raw_df = pd.DataFrame(all_records)
        return raw_df, file_metrics

    def run_pipeline(
        self,
        max_files: Optional[int] = None,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Execute the end-to-end ingestion, cleaning, detection, and storage pipeline.

        Parameters
        ----------
        max_files : Optional[int], optional
            Limit the number of input files to process.
        save_to_db : bool, optional
            Whether to commit cleaned pings and anomalies to SQLite. Defaults to True.

        Returns
        -------
        Dict[str, Any]
            Comprehensive pipeline execution audit report.
        """
        start_time = time.perf_counter()
        logger.info("=== Starting Telematics Ingestion Pipeline ===")

        # 1. Ingest raw records from JSON files
        raw_df, file_metrics = self.ingest_raw_dataset(max_files=max_files)
        raw_pings_count = len(raw_df)

        if raw_pings_count == 0:
            logger.warning("No records ingested. Pipeline terminating early.")
            return {
                **file_metrics,
                "clean_pings_saved": 0,
                "anomalies_detected": 0,
                "elapsed_seconds": round(time.perf_counter() - start_time, 2),
            }

        # 2. Vectorized cleaning and precomputations (TelematicsCleaner)
        logger.info("Running TelematicsCleaner on %d raw records...", raw_pings_count)
        cleaned_df, cleaning_audit = self.cleaner.clean_data(raw_df)
        clean_pings_count = len(cleaned_df)

        # 3. Anomaly Detection
        logger.info("Executing multi-condition anomaly detectors...")
        cleaned_df, fuel_anomalies = self.fuel_detector.detect(cleaned_df)
        cleaned_df, moon_anomalies = self.moonlight_detector.detect(cleaned_df)
        cleaned_df, kin_anomalies = self.kinematic_detector.detect(cleaned_df)

        all_anomalies: List[AnomalyEvent] = fuel_anomalies + moon_anomalies + kin_anomalies
        total_anomalies_count = len(all_anomalies)

        # Anomaly categorization summary
        anomaly_breakdown: Dict[str, int] = {}
        for anom in all_anomalies:
            anomaly_breakdown[anom.incident_type] = anomaly_breakdown.get(anom.incident_type, 0) + 1

        # 4. Database Persistence
        pings_persisted = 0
        anomalies_persisted = 0
        if save_to_db and clean_pings_count > 0:
            logger.info("Persisting %d cleaned pings and %d anomalies to SQLite...", clean_pings_count, total_anomalies_count)
            pings_persisted = self.db_manager.save_telematics_pings(cleaned_df)
            anomalies_persisted = self.db_manager.log_anomalies_bulk(all_anomalies)

        elapsed_seconds = round(time.perf_counter() - start_time, 2)

        pipeline_report: Dict[str, Any] = {
            "total_files_scanned": file_metrics["total_files_scanned"],
            "files_populated": file_metrics["files_populated"],
            "files_empty_or_malformed": file_metrics["files_empty"] + file_metrics["files_malformed"],
            "raw_pings_ingested": raw_pings_count,
            "valid_clean_pings": clean_pings_count,
            "pings_persisted_to_db": pings_persisted,
            "anomalies_detected": total_anomalies_count,
            "anomalies_persisted_to_db": anomalies_persisted,
            "anomaly_breakdown": anomaly_breakdown,
            "data_retention_rate_pct": cleaning_audit["data_retention_rate_pct"],
            "dropped_invalid_coordinates": cleaning_audit["dropped_invalid_coordinates"],
            "dropped_invalid_speed": cleaning_audit["dropped_invalid_speed"],
            "dropped_invalid_fuel": cleaning_audit["dropped_invalid_fuel"],
            "dropped_duplicate_timestamps": cleaning_audit["dropped_duplicate_timestamps"],
            "unique_vehicles_ingested": int(cleaned_df["vehicle_id"].nunique()) if clean_pings_count > 0 else 0,
            "database_file": str(self.db_manager.db_path),
            "elapsed_seconds": elapsed_seconds,
        }

        # 5. Output Terminal Summary Report
        self.print_summary_report(pipeline_report)
        return pipeline_report

    @staticmethod
    def print_summary_report(metrics: Dict[str, Any]) -> None:
        """Format and print an executive terminal report of the ingestion run."""
        separator = "=" * 76
        sub_sep = "-" * 76

        print("\n" + separator)
        print("          INNONICS FLEET TELEMATICS INGESTION & AUDIT REPORT          ")
        print(separator)
        print(" 1. FILE DISCOVERY & EXTRACTION")
        print(f"    - Total Files Scanned        : {metrics.get('total_files_scanned', 0):,}")
        print(f"    - Populated Data Files       : {metrics.get('files_populated', 0):,}")
        print(f"    - Empty / Malformed Skipped  : {metrics.get('files_empty_or_malformed', 0):,}")
        print(f"    - Raw Telematics Pings       : {metrics.get('raw_pings_ingested', 0):,}")
        print(sub_sep)
        print(" 2. DATA QUALITY & CLEANING (TelematicsCleaner)")
        print(f"    - Valid Clean Pings          : {metrics.get('valid_clean_pings', 0):,}")
        print(f"    - Data Retention Rate        : {metrics.get('data_retention_rate_pct', 0.0):.2f}%")
        print(f"    - Dropped Invalid Coordinates: {metrics.get('dropped_invalid_coordinates', 0):,}")
        print(f"    - Dropped Invalid Speed      : {metrics.get('dropped_invalid_speed', 0):,}")
        print(f"    - Dropped Invalid Fuel       : {metrics.get('dropped_invalid_fuel', 0):,}")
        print(f"    - Dropped Duplicate Pings    : {metrics.get('dropped_duplicate_timestamps', 0):,}")
        print(f"    - Unique Vehicles Processed  : {metrics.get('unique_vehicles_ingested', 0):,}")
        print(sub_sep)
        print(" 3. ANOMALY DETECTION BREAKDOWN")
        print(f"    - Total Anomalies Detected   : {metrics.get('anomalies_detected', 0):,}")
        breakdown = metrics.get("anomaly_breakdown", {})
        if breakdown:
            for incident_type, count in sorted(breakdown.items(), key=lambda x: -x[1]):
                print(f"      * {incident_type:<26}: {count:,}")
        else:
            print("      * No operational anomalies detected.")
        print(sub_sep)
        print(" 4. DATABASE PERSISTENCE STATUS")
        print(f"    - Pings Persisted in DB      : {metrics.get('pings_persisted_to_db', 0):,}")
        print(f"    - Anomalies Logged in DB     : {metrics.get('anomalies_persisted_to_db', 0):,}")
        print(f"    - Storage Target             : {metrics.get('database_file', 'N/A')}")
        print(f"    - Execution Window           : {metrics.get('elapsed_seconds', 0.0):.2f} seconds")
        print(separator + "\n")


def run_ingestion(
    data_dir: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    max_files: Optional[int] = None,
    save_to_db: bool = True,
) -> Dict[str, Any]:
    """Convenience functional interface to trigger the ingestion pipeline."""
    pipeline = IngestionPipeline(data_dir=data_dir, db_path=db_path)
    return pipeline.run_pipeline(max_files=max_files, save_to_db=save_to_db)


def main() -> None:
    """Command-line interface entry point for telematics ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest, clean, analyze, and persist Innonics IoT telematics dataset into SQLite."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to directory containing telematics JSON files (default: data/)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database file (default: database/telematics_core.db)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of JSON files to ingest (default: all)",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Run ingestion and anomaly analysis in dry-run mode without saving to SQLite",
    )

    args = parser.parse_args()
    run_ingestion(
        data_dir=args.data_dir,
        db_path=args.db_path,
        max_files=args.max_files,
        save_to_db=not args.skip_db,
    )


if __name__ == "__main__":
    main()
