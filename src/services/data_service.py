"""Data Service Layer for Telematics Streamlit Dashboard.

Provides high-performance, cached, parameterised database queries,
safety score computations, telemetry freshness evaluation, and CRUD operations.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

from database.db_manager import DEFAULT_DB_PATH, TelematicsDBManager
from src.scoring import RiskScoreReport, RiskScoringEngine, RiskTier
from src.services.telegram_service import TelegramNotificationService
from src.utils.timezone import (
    MYT_TZ,
    is_ghost_hours_myt,
    now_myt,
    to_myt_datetime,
    to_myt_str,
)

logger = logging.getLogger(__name__)


class TelematicsDataService:
    """Service abstraction connecting presentation layer to database and analytics engines."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        auto_seed: bool = False,
    ) -> None:
        """Initialize data service with database manager."""
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.db = TelematicsDBManager(db_path=self.db_path, auto_seed=auto_seed)
        self.scoring_engine = RiskScoringEngine()
        self.telegram_service = TelegramNotificationService()

    # --------------------------------------------------------------------------
    # Database Connection & Health
    # --------------------------------------------------------------------------

    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and verify presence of schema tables."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [r[0] for r in cursor.fetchall()]
            return {"status": "HEALTHY", "tables": tables, "db_path": str(self.db_path)}
        except Exception as exc:
            logger.error("Database connection failure: %s", exc)
            return {"status": "ERROR", "error": str(exc), "db_path": str(self.db_path)}

    def _get_utc_interval_for_date_and_shift(
        self,
        target_date: Optional[Union[str, date]] = None,
        shift_window: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Convert a calendar date in MYT and shift window to UTC timestamp bounds."""
        if not target_date:
            return None, None

        if isinstance(target_date, str):
            try:
                base_date = datetime.strptime(target_date[:10], "%Y-%m-%d").date()
            except Exception:
                return None, None
        else:
            base_date = target_date

        shift_key = (shift_window or "").lower()
        if "day" in shift_key:
            # 08:00 - 17:00 MYT -> 00:00 - 09:00 UTC on base_date
            utc_start = datetime(base_date.year, base_date.month, base_date.day, 0, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
            utc_end = datetime(base_date.year, base_date.month, base_date.day, 9, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
        elif "ghost" in shift_key or "night" in shift_key:
            # Ghost Hours: 22:00 - 04:59 MYT -> 14:00 - 20:59 UTC
            utc_start = datetime(base_date.year, base_date.month, base_date.day, 14, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
            utc_end = datetime(base_date.year, base_date.month, base_date.day, 20, 59, 59).strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Full 24h shift: 00:00:00 - 23:59:59 MYT
            # In UTC: (base_date - 1 day) 16:00:00 to base_date 15:59:59
            start_myt = datetime(base_date.year, base_date.month, base_date.day, 0, 0, 0, tzinfo=MYT_TZ)
            end_myt = datetime(base_date.year, base_date.month, base_date.day, 23, 59, 59, tzinfo=MYT_TZ)
            utc_start = start_myt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            utc_end = end_myt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        return utc_start, utc_end

    def get_available_historical_dates(self) -> List[str]:
        """Fetch distinct calendar dates (MYT) available in the telematics database archive."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT timestamp FROM telematics_pings;")
            all_ts = [r[0] for r in cursor.fetchall() if r[0]]

        if not all_ts:
            return ["2024-08-09", "2024-01-10"]

        myt_dates = sorted(list(set(to_myt_datetime(ts).strftime("%Y-%m-%d") for ts in all_ts)))
        return myt_dates

    # --------------------------------------------------------------------------
    # Fleet Summary & Freshness Metrics
    # --------------------------------------------------------------------------

    def get_fleet_kpis(
        self,
        freshness_hours: float = 24.0,
        freshness_mode: str = "relative_to_dataset",
    ) -> Dict[str, Any]:
        """Compute high-level fleet operational KPIs from database.

        Parameters
        ----------
        freshness_hours : float
            Cutoff window in hours for considering a vehicle active.
        freshness_mode : str
            'relative_to_dataset' (relative to newest ping in DB) or 'wall_clock_now'.

        Returns
        -------
        Dict[str, Any]
            Fleet KPI metrics.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Total vehicles
            cursor.execute("SELECT COUNT(*) FROM vehicles;")
            total_vehicles = cursor.fetchone()[0]

            # 2. Total pings
            cursor.execute("SELECT COUNT(*) FROM telematics_pings;")
            total_pings = cursor.fetchone()[0]

            # 3. Unresolved anomalies
            cursor.execute("SELECT COUNT(*) FROM anomaly_logs WHERE status != 'RESOLVED';")
            unresolved_anomalies = cursor.fetchone()[0]

            # 4. Total logged anomalies
            cursor.execute("SELECT COUNT(*) FROM anomaly_logs;")
            total_anomalies = cursor.fetchone()[0]

            # 5. Latest ping timestamp across entire dataset
            cursor.execute("SELECT MAX(timestamp) FROM telematics_pings;")
            max_ts_str = cursor.fetchone()[0]

        # Determine reference time for freshness
        if max_ts_str and freshness_mode == "relative_to_dataset":
            try:
                reference_time = pd.to_datetime(max_ts_str)
            except Exception:
                reference_time = datetime.now(timezone.utc)
        else:
            reference_time = datetime.now(timezone.utc)

        # 6. Active vehicles count
        active_vehicles = 0
        fleet_safety_scores: List[float] = []

        vehicles_df = self.get_fleet_status_table(
            freshness_hours=freshness_hours,
            freshness_mode=freshness_mode,
        )

        if not vehicles_df.empty:
            active_vehicles = int((vehicles_df["freshness_status"] == "Active").sum())
            valid_scores = vehicles_df["safety_score"].dropna()
            fleet_safety_scores = valid_scores.tolist()

        avg_safety_score = (
            round(float(np.mean(fleet_safety_scores)), 2)
            if fleet_safety_scores
            else 100.00
        )

        return {
            "total_vehicles": total_vehicles,
            "active_vehicles": active_vehicles,
            "total_pings": total_pings,
            "unresolved_anomalies": unresolved_anomalies,
            "total_anomalies": total_anomalies,
            "average_safety_score": avg_safety_score,
            "latest_ping_time": max_ts_str,
            "freshness_rule": f"Active if reported within {freshness_hours}h ({freshness_mode})",
            "scored_vehicles_count": len(fleet_safety_scores),
        }

    def get_fleet_status_table(
        self,
        freshness_hours: float = 24.0,
        freshness_mode: str = "relative_to_dataset",
    ) -> pd.DataFrame:
        """Fetch consolidated fleet status table with latest ping and safety score per vehicle.

        Parameters
        ----------
        freshness_hours : float
            Threshold hours for freshness.
        freshness_mode : str
            Freshness evaluation mode.

        Returns
        -------
        pd.DataFrame
            Table containing vehicle specs, latest metrics, freshness, and safety scores.
        """
        # Fetch all vehicles
        vehicles = self.db.get_all_vehicles()
        if not vehicles:
            return pd.DataFrame()

        v_df = pd.DataFrame(vehicles)

        # Fetch latest ping per vehicle using SQL subquery
        query_latest = """
        SELECT p.vehicle_id, p.timestamp, p.speed, p.fuel_level, p.ignition, p.latitude, p.longitude, p.heading
        FROM telematics_pings p
        INNER JOIN (
            SELECT vehicle_id, MAX(timestamp) AS max_ts
            FROM telematics_pings
            GROUP BY vehicle_id
        ) latest ON p.vehicle_id = latest.vehicle_id AND p.timestamp = latest.max_ts;
        """

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_latest)
            latest_rows = [dict(r) for r in cursor.fetchall()]

        latest_df = pd.DataFrame(latest_rows) if latest_rows else pd.DataFrame()

        # Merge vehicles with latest telemetry
        if not latest_df.empty:
            merged = pd.merge(v_df, latest_df, on="vehicle_id", how="left")
        else:
            merged = v_df.copy()
            for col in ["timestamp", "speed", "fuel_level", "ignition", "latitude", "longitude", "heading"]:
                merged[col] = None

        # Reference time for freshness
        max_ts_str = latest_df["timestamp"].max() if not latest_df.empty else None
        if max_ts_str and freshness_mode == "relative_to_dataset":
            try:
                reference_time = pd.to_datetime(max_ts_str)
            except Exception:
                reference_time = pd.Timestamp.now(tz="UTC").tz_localize(None)
        else:
            reference_time = pd.Timestamp.now(tz="UTC").tz_localize(None)

        # Compute freshness and scores
        freshness_status = []
        scores = []
        tiers = []

        # Pre-fetch anomalies count per vehicle
        anomaly_counts_by_vehicle = self.get_anomaly_counts_by_vehicle()

        for _, row in merged.iterrows():
            vid = row["vehicle_id"]
            ts = row.get("timestamp")

            if pd.notna(ts):
                try:
                    ping_time = pd.to_datetime(ts)
                    # Compute time delta in hours
                    diff_hours = abs((reference_time - ping_time).total_seconds()) / 3600.0
                    if diff_hours <= freshness_hours:
                        f_status = "Active"
                    elif diff_hours <= freshness_hours * 3.0:
                        f_status = "Recent"
                    else:
                        f_status = "Stale"
                except Exception:
                    f_status = "Stale"
            else:
                f_status = "No Telemetry"

            freshness_status.append(f_status)

            # Score calculation
            anom_counts = anomaly_counts_by_vehicle.get(vid, {})
            score_rep = self.scoring_engine.calculate_score(anom_counts, vehicle_id=vid)
            scores.append(score_rep.safety_score)
            tiers.append(score_rep.risk_tier)

        merged["freshness_status"] = freshness_status
        merged["safety_score"] = scores
        merged["risk_tier"] = tiers

        # Format ignition
        def format_ignition(val: Any) -> str:
            if pd.isna(val):
                return "Unknown"
            if str(val) in ("1", "ON", "True", "1.0"):
                return "ON"
            return "OFF"

        merged["ignition_label"] = merged["ignition"].apply(format_ignition)
        return merged

    # --------------------------------------------------------------------------
    # Telemetry Time Series Queries
    # --------------------------------------------------------------------------

    def get_vehicle_telemetry(
        self,
        vehicle_id: str,
        limit: int = 5000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_date: Optional[Union[str, date]] = None,
        shift_window: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch chronologically sorted telemetry records for a specific vehicle.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier.
        limit : int, optional
            Limit max rows to protect UI performance. Defaults to 5000.
        start_date : Optional[str], optional
            ISO date string filter (e.g. '2024-01-01').
        end_date : Optional[str], optional
            ISO date string filter (e.g. '2024-12-31').
        target_date : Optional[Union[str, date]], optional
            Calendar date filter in MYT.
        shift_window : Optional[str], optional
            Shift window filter.

        Returns
        -------
        pd.DataFrame
            Telemetry records.
        """
        clauses = ["vehicle_id = ?"]
        params: List[Any] = [str(vehicle_id)]

        if target_date:
            utc_start, utc_end = self._get_utc_interval_for_date_and_shift(target_date, shift_window)
            if utc_start and utc_end:
                clauses.append("timestamp >= ?")
                params.append(utc_start)
                clauses.append("timestamp <= ?")
                params.append(utc_end)
        else:
            if start_date:
                clauses.append("timestamp >= ?")
                params.append(f"{start_date} 00:00:00")
            if end_date:
                clauses.append("timestamp <= ?")
                params.append(f"{end_date} 23:59:59")

        where_sql = "WHERE " + " AND ".join(clauses)
        query = f"""
        SELECT id, vehicle_id, timestamp, latitude, longitude, speed, fuel_level,
               heading, ignition, delta_time, delta_speed, delta_fuel
        FROM telematics_pings
        {where_sql}
        ORDER BY timestamp ASC
        LIMIT ?;
        """
        params.append(int(limit))

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # --------------------------------------------------------------------------
    # Anomaly Logs Queries
    # --------------------------------------------------------------------------

    def get_anomalies(
        self,
        vehicle_id: Optional[str] = None,
        status: Optional[str] = None,
        incident_type: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Query anomaly incident logs with multi-parameter filtering.

        Parameters
        ----------
        vehicle_id : Optional[str], optional
            Filter by vehicle.
        status : Optional[str], optional
            Filter by status ('OPEN', 'RESOLVED', etc.).
        incident_type : Optional[str], optional
            Filter by incident type.
        limit : int, optional
            Maximum records. Defaults to 1000.

        Returns
        -------
        pd.DataFrame
            Tabular anomaly logs.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if vehicle_id:
            clauses.append("vehicle_id = ?")
            params.append(str(vehicle_id))
        if status and status.upper() != "ALL":
            clauses.append("status = ?")
            params.append(str(status).upper())
        if incident_type and incident_type.upper() != "ALL":
            clauses.append("incident_type = ?")
            params.append(str(incident_type).upper())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
        SELECT id, vehicle_id, timestamp, incident_type, magnitude, status
        FROM anomaly_logs
        {where_sql}
        ORDER BY id DESC
        LIMIT ?;
        """
        params.append(int(limit))

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df

    def get_anomaly_counts_by_vehicle(
        self,
        target_date: Optional[Union[str, date]] = None,
        shift_window: Optional[str] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Compute incident counts per category for every vehicle in database."""
        utc_start, utc_end = self._get_utc_interval_for_date_and_shift(target_date, shift_window)
        where_sql = ""
        params: List[Any] = []
        if utc_start and utc_end:
            where_sql = "WHERE timestamp >= ? AND timestamp <= ?"
            params = [utc_start, utc_end]

        query = f"""
        SELECT vehicle_id, incident_type, COUNT(*) AS cnt
        FROM anomaly_logs
        {where_sql}
        GROUP BY vehicle_id, incident_type;
        """
        counts_map: Dict[str, Dict[str, int]] = {}

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            for row in cursor.fetchall():
                vid = row["vehicle_id"]
                itype = row["incident_type"]
                cnt = row["cnt"]
                if vid not in counts_map:
                    counts_map[vid] = {}

                # Map to scoring categories
                cat = self.scoring_engine.INCIDENT_TYPE_MAP.get(itype.upper())
                if cat:
                    counts_map[vid][cat] = counts_map[vid].get(cat, 0) + cnt

        return counts_map

    # --------------------------------------------------------------------------
    # Risk Scoring & Baseline vs. FYP2 Comparison
    # --------------------------------------------------------------------------

    def calculate_vehicle_safety_score(
        self,
        vehicle_id: str,
        custom_weights: Optional[Dict[str, float]] = None,
        compound_multiplier: float = 1.5,
    ) -> RiskScoreReport:
        """Calculate safety score and explainable breakdown for a specific vehicle.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier.
        custom_weights : Optional[Dict[str, float]], optional
            Dynamic weight overrides.
        compound_multiplier : float, optional
            Interaction compounding multiplier. Defaults to 1.5.

        Returns
        -------
        RiskScoreReport
            Explainable audit report.
        """
        counts_map = self.get_anomaly_counts_by_vehicle()
        vehicle_counts = counts_map.get(str(vehicle_id), {})

        engine = RiskScoringEngine(
            weights=custom_weights,
            compound_multiplier=compound_multiplier,
        )
        return engine.calculate_score(vehicle_counts, vehicle_id=str(vehicle_id))

    def compare_fyp1_vs_fyp2_scoring(self, vehicle_id: str) -> Dict[str, Any]:
        """Compare FYP1 naive linear scoring against FYP2 non-linear compounding model.

        FYP1 Model: Linear subtraction without co-occurrence interaction multiplier.
        FYP2 Model: Non-linear compounding state-machine model (1.5x interaction penalty).

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier.

        Returns
        -------
        Dict[str, Any]
            Comparative metrics between baseline and advanced model.
        """
        counts_map = self.get_anomaly_counts_by_vehicle()
        counts = counts_map.get(str(vehicle_id), {})

        # 1. FYP1 Baseline (Linear: compound_multiplier = 1.0)
        fyp1_engine = RiskScoringEngine(compound_multiplier=1.0)
        fyp1_report = fyp1_engine.calculate_score(counts, vehicle_id=str(vehicle_id))

        # 2. FYP2 Advanced (Non-linear compounding: compound_multiplier = 1.5)
        fyp2_engine = RiskScoringEngine(compound_multiplier=1.5)
        fyp2_report = fyp2_engine.calculate_score(counts, vehicle_id=str(vehicle_id))

        return {
            "vehicle_id": vehicle_id,
            "fyp1_score": fyp1_report.safety_score,
            "fyp1_tier": fyp1_report.risk_tier,
            "fyp1_deductions": fyp1_report.total_deductions,
            "fyp2_score": fyp2_report.safety_score,
            "fyp2_tier": fyp2_report.risk_tier,
            "fyp2_deductions": fyp2_report.total_deductions,
            "compounding_applied": fyp2_report.compounding_applied,
            "compounding_interactions": fyp2_report.compounded_interactions,
            "delta_points": round(fyp1_report.safety_score - fyp2_report.safety_score, 2),
        }

    # --------------------------------------------------------------------------
    # Administrative CRUD Operations & Audit Logging
    # --------------------------------------------------------------------------

    def register_vehicle(
        self,
        vehicle_id: str,
        plate_number: str,
        model: str,
        fuel_capacity_liters: float,
    ) -> Tuple[bool, str]:
        """Register a new commercial vehicle with input validation."""
        v_id = str(vehicle_id).strip()
        plate = str(plate_number).strip().upper()
        mdl = str(model).strip()

        if not v_id:
            return False, "Vehicle ID cannot be empty."
        if not plate:
            return False, "Plate number cannot be empty."
        if not mdl:
            return False, "Model cannot be empty."
        if fuel_capacity_liters <= 0:
            return False, "Fuel capacity must be greater than 0 Liters."

        success = self.db.add_vehicle(v_id, plate, mdl, fuel_capacity_liters)
        if success:
            self.db.log_admin_action(
                action="CREATE",
                entity_type="VEHICLE",
                entity_id=v_id,
                details=f"Created vehicle {v_id} ({plate}, {mdl}, {fuel_capacity_liters}L)",
            )
            return True, f"Vehicle '{v_id}' registered successfully."
        return False, f"Vehicle '{v_id}' already exists in registry."

    def update_vehicle(
        self,
        vehicle_id: str,
        plate_number: str,
        model: str,
        fuel_capacity_liters: float,
    ) -> Tuple[bool, str]:
        """Update existing vehicle registration attributes."""
        plate = str(plate_number).strip().upper()
        mdl = str(model).strip()

        if not plate:
            return False, "Plate number cannot be empty."
        if not mdl:
            return False, "Model cannot be empty."
        if fuel_capacity_liters <= 0:
            return False, "Fuel capacity must be greater than 0 Liters."

        success = self.db.update_vehicle(vehicle_id, plate, mdl, fuel_capacity_liters)
        if success:
            return True, f"Vehicle '{vehicle_id}' updated successfully."
        return False, f"Vehicle '{vehicle_id}' not found."

    def delete_vehicle(self, vehicle_id: str) -> Tuple[bool, str]:
        """Delete a vehicle and cascade delete related telemetry and logs."""
        success = self.db.delete_vehicle(vehicle_id)
        if success:
            self.db.log_admin_action(
                action="DELETE",
                entity_type="VEHICLE",
                entity_id=vehicle_id,
                details=f"Deleted vehicle {vehicle_id} and cascaded telemetry/logs",
            )
            return True, f"Vehicle '{vehicle_id}' deleted successfully."
        return False, f"Vehicle '{vehicle_id}' not found."

    def add_driver_schedule(
        self,
        driver_id: str,
        vehicle_id: str,
        shift_start: str,
        shift_end: str,
        is_night_shift: bool,
    ) -> Tuple[bool, str]:
        """Insert a driver schedule roster with foreign key verification."""
        d_id = str(driver_id).strip().upper()
        v_id = str(vehicle_id).strip()

        if not d_id:
            return False, "Driver ID cannot be empty."
        if not v_id:
            return False, "Vehicle must be selected."

        try:
            sched_id = self.db.add_driver_schedule(
                driver_id=d_id,
                vehicle_id=v_id,
                shift_start=str(shift_start),
                shift_end=str(shift_end),
                is_night_shift=is_night_shift,
            )
            self.db.log_admin_action(
                action="CREATE",
                entity_type="DRIVER_SCHEDULE",
                entity_id=str(sched_id),
                details=f"Created schedule for {d_id} on {v_id} ({shift_start}-{shift_end}, NightShift: {is_night_shift})",
            )
            return True, f"Schedule #{sched_id} created successfully."
        except ValueError as exc:
            return False, str(exc)

    def update_driver_schedule(
        self,
        schedule_id: int,
        driver_id: str,
        vehicle_id: str,
        shift_start: str,
        shift_end: str,
        is_night_shift: bool,
    ) -> Tuple[bool, str]:
        """Update an existing driver shift schedule."""
        success = self.db.update_driver_schedule(
            schedule_id=schedule_id,
            driver_id=driver_id,
            vehicle_id=vehicle_id,
            shift_start=shift_start,
            shift_end=shift_end,
            is_night_shift=is_night_shift,
        )
        if success:
            return True, f"Schedule #{schedule_id} updated successfully."
        return False, f"Schedule #{schedule_id} not found."

    def delete_driver_schedule(self, schedule_id: int) -> Tuple[bool, str]:
        """Delete a driver shift schedule by ID."""
        success = self.db.delete_driver_schedule(schedule_id)
        if success:
            self.db.log_admin_action(
                action="DELETE",
                entity_type="DRIVER_SCHEDULE",
                entity_id=str(schedule_id),
                details=f"Deleted schedule #{schedule_id}",
            )
            return True, f"Schedule #{schedule_id} deleted successfully."
        return False, f"Schedule #{schedule_id} not found."

    def resolve_anomaly(
        self,
        anomaly_id: int,
        resolution_notes: str,
        user_id: str = "admin",
    ) -> Tuple[bool, str]:
        """Resolve an anomaly record with administrative justification."""
        notes = str(resolution_notes).strip()
        if not notes:
            return False, "Resolution justification rationale is required."

        success = self.db.update_anomaly_with_resolution(
            anomaly_id=anomaly_id,
            status="RESOLVED",
            resolution_notes=notes,
            user_id=user_id,
        )
        if success:
            return True, f"Anomaly #{anomaly_id} marked as RESOLVED."
        return False, f"Anomaly #{anomaly_id} not found."

    def get_audit_trail(self, limit: int = 100) -> pd.DataFrame:
        """Fetch administrative audit trail logs."""
        logs = self.db.get_admin_audit_logs(limit=limit)
        return pd.DataFrame(logs) if logs else pd.DataFrame()

    # --------------------------------------------------------------------------
    # Forensic Audit & Leaderboard Operations
    # --------------------------------------------------------------------------

    def get_forensic_leaderboard(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
        compound_multiplier: float = 1.5,
        target_date: Optional[Union[str, date]] = None,
        shift_window: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute comprehensive fleet leaderboard for forensic audit.

        Ranks drivers and vehicles based on historical shift accumulation, total fuel loss,
        anomaly infractions, and dynamic MCDA-calculated Driver Safety Scores (DSS).

        Parameters
        ----------
        custom_weights : Optional[Dict[str, float]], optional
            Dynamic MCDA weight overrides.
        compound_multiplier : float, optional
            Interaction compounding multiplier. Defaults to 1.5.
        target_date : Optional[Union[str, date]], optional
            Calendar date filter in MYT.
        shift_window : Optional[str], optional
            Shift window filter.

        Returns
        -------
        pd.DataFrame
            Leaderboard DataFrame sorted by safety score descending with rank.
        """
        vehicles = self.db.get_all_vehicles()
        if not vehicles:
            return pd.DataFrame()

        engine = RiskScoringEngine(
            weights=custom_weights,
            compound_multiplier=compound_multiplier,
        )
        counts_map = self.get_anomaly_counts_by_vehicle(target_date=target_date, shift_window=shift_window)

        utc_start, utc_end = self._get_utc_interval_for_date_and_shift(target_date, shift_window)
        anom_where = ""
        ping_where = ""
        filter_params: List[Any] = []

        if utc_start and utc_end:
            anom_where = "WHERE timestamp >= ? AND timestamp <= ?"
            ping_where = "WHERE timestamp >= ? AND timestamp <= ?"
            filter_params = [utc_start, utc_end]

        # Query aggregate anomaly metrics per vehicle
        query_anoms = f"""
        SELECT vehicle_id,
               COUNT(*) AS total_anomalies,
               SUM(CASE WHEN incident_type IN ('FUEL_THEFT', 'SUDDEN_FUEL_DROP') THEN ABS(magnitude) ELSE 0.0 END) AS fuel_loss_liters,
               SUM(CASE WHEN incident_type = 'REFUEL' THEN magnitude ELSE 0.0 END) AS refuel_liters,
               SUM(CASE WHEN incident_type = 'GEOFENCE_BREACH' THEN 1 ELSE 0 END) AS geofence_breaches,
               SUM(CASE WHEN incident_type = 'UNAUTHORIZED_NIGHT_DRIVING' THEN 1 ELSE 0 END) AS night_breaches,
               SUM(CASE WHEN incident_type = 'OVERSPEEDING' THEN 1 ELSE 0 END) AS overspeed_count,
               SUM(CASE WHEN incident_type = 'EXCESSIVE_IDLING' THEN 1 ELSE 0 END) AS idling_count
        FROM anomaly_logs
        {anom_where}
        GROUP BY vehicle_id;
        """

        # Query aggregate ping metrics per vehicle
        query_pings = f"""
        SELECT vehicle_id,
               COUNT(*) AS total_pings,
               MIN(timestamp) AS shift_start_utc,
               MAX(timestamp) AS shift_end_utc,
               MAX(speed) AS max_speed,
               AVG(speed) AS avg_speed
        FROM telematics_pings
        {ping_where}
        GROUP BY vehicle_id;
        """

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_anoms, tuple(filter_params))
            anom_agg = {r["vehicle_id"]: dict(r) for r in cursor.fetchall()}

            cursor.execute(query_pings, tuple(filter_params))
            ping_agg = {r["vehicle_id"]: dict(r) for r in cursor.fetchall()}

        leaderboard_rows: List[Dict[str, Any]] = []

        for v in vehicles:
            vid = v["vehicle_id"]
            v_anom = anom_agg.get(vid, {})
            v_ping = ping_agg.get(vid, {})

            # Compute score via scoring engine
            vehicle_counts = counts_map.get(vid, {})
            score_rep = engine.calculate_score(vehicle_counts, vehicle_id=vid)

            fuel_loss = round(float(v_anom.get("fuel_loss_liters", 0.0)), 2)
            refuels = round(float(v_anom.get("refuel_liters", 0.0)), 2)
            tot_anoms = int(v_anom.get("total_anomalies", 0))
            geofence = int(v_anom.get("geofence_breaches", 0))
            night = int(v_anom.get("night_breaches", 0))
            overspeed = int(v_anom.get("overspeed_count", 0))
            idling = int(v_anom.get("idling_count", 0))

            pings_count = int(v_ping.get("total_pings", 0))
            max_spd = round(float(v_ping.get("max_speed", 0.0) or 0.0), 1)
            avg_spd = round(float(v_ping.get("avg_speed", 0.0) or 0.0), 1)

            # When auditing a specific date/shift, skip vehicles with zero shift activity
            if (target_date or shift_window) and pings_count == 0 and tot_anoms == 0:
                continue

            shift_start_myt = to_myt_str(v_ping.get("shift_start_utc"))
            shift_end_myt = to_myt_str(v_ping.get("shift_end_utc"))

            # Critical incidents count (Fuel theft + night breaches + overspeeding)
            critical_incidents = int(vehicle_counts.get("fuel_theft", 0)) + night + overspeed

            leaderboard_rows.append({
                "vehicle_id": vid,
                "plate_number": v.get("plate_number", ""),
                "model": v.get("model", ""),
                "fuel_capacity_liters": v.get("fuel_capacity_liters", 0.0),
                "safety_score": score_rep.safety_score,
                "risk_tier": score_rep.risk_tier,
                "total_deductions": score_rep.total_deductions,
                "compounding_applied": score_rep.compounding_applied,
                "fuel_loss_liters": fuel_loss,
                "refuel_liters": refuels,
                "critical_incidents": critical_incidents,
                "geofence_breaches": geofence,
                "night_breaches": night,
                "overspeed_count": overspeed,
                "idling_count": idling,
                "total_anomalies": tot_anoms,
                "total_pings": pings_count,
                "max_speed": max_spd,
                "avg_speed": avg_spd,
                "shift_start_myt": shift_start_myt,
                "shift_end_myt": shift_end_myt,
            })

        df = pd.DataFrame(leaderboard_rows)
        if df.empty:
            return df

        # Sort: Highest safety score first, then lowest fuel loss
        df = df.sort_values(by=["safety_score", "fuel_loss_liters"], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def get_fuel_loss_records(
        self,
        vehicle_id: Optional[str] = None,
        limit: int = 500,
        target_date: Optional[Union[str, date]] = None,
        shift_window: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical fuel loss and sudden drop incident logs formatted in MYT.

        Parameters
        ----------
        vehicle_id : Optional[str], optional
            Filter for specific vehicle. Defaults to None.
        limit : int, optional
            Maximum records. Defaults to 500.
        target_date : Optional[Union[str, date]], optional
            Calendar date filter in MYT.
        shift_window : Optional[str], optional
            Shift window filter.

        Returns
        -------
        pd.DataFrame
            Fuel loss logs with MYT timestamps.
        """
        clauses = ["incident_type IN ('FUEL_THEFT', 'SUDDEN_FUEL_DROP')"]
        params: List[Any] = []

        if vehicle_id and vehicle_id.upper() != "ALL":
            clauses.append("vehicle_id = ?")
            params.append(str(vehicle_id))

        utc_start, utc_end = self._get_utc_interval_for_date_and_shift(target_date, shift_window)
        if utc_start and utc_end:
            clauses.append("timestamp >= ?")
            params.append(utc_start)
            clauses.append("timestamp <= ?")
            params.append(utc_end)

        where_sql = f"WHERE {' AND '.join(clauses)}"
        query = f"""
        SELECT id, vehicle_id, timestamp, incident_type, magnitude, status
        FROM anomaly_logs
        {where_sql}
        ORDER BY id DESC
        LIMIT ?;
        """
        params.append(int(limit))

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["timestamp_myt"] = df["timestamp"].apply(to_myt_str)
        df["magnitude_liters"] = df["magnitude"].apply(lambda m: abs(float(m)))
        return df

    def get_radar_profile_metrics(
        self,
        vehicle_id: str,
        custom_weights: Optional[Dict[str, float]] = None,
        compound_multiplier: float = 1.5,
        target_date: Optional[Union[str, date]] = None,
        shift_window: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute 5-axis explainable risk profile for audited vehicle vs fleet baseline.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier.
        custom_weights : Optional[Dict[str, float]], optional
            Dynamic MCDA weights.
        compound_multiplier : float, optional
            Interaction compounding multiplier. Defaults to 1.5.
        target_date : Optional[Union[str, date]], optional
            Calendar date filter in MYT.
        shift_window : Optional[str], optional
            Shift window filter.

        Returns
        -------
        Dict[str, Any]
            Radar profile scores, fleet average baseline, and audit report.
        """
        engine = RiskScoringEngine(
            weights=custom_weights,
            compound_multiplier=compound_multiplier,
        )
        counts_map = self.get_anomaly_counts_by_vehicle(target_date=target_date, shift_window=shift_window)
        vehicle_counts = counts_map.get(str(vehicle_id), {})
        vehicle_rep = engine.calculate_score(vehicle_counts, vehicle_id=str(vehicle_id))

        # Normalized domain scoring helper (0 to 100 compliance scale)
        def _score_domain(deduction: float, scale_factor: float = 1.0) -> float:
            return round(max(0.0, min(100.0, 100.0 - (deduction * scale_factor))), 1)

        v_breakdown = vehicle_rep.breakdown
        v_scores = {
            "Fuel Integrity": _score_domain(v_breakdown.get("fuel_theft", 0.0), 2.0),
            "Route Compliance": _score_domain(v_breakdown.get("moonlighting", 0.0), 0.5),
            "Kinematic Smoothness": _score_domain(v_breakdown.get("harsh_driving", 0.0), 5.0),
            "Speed Adherence": _score_domain(v_breakdown.get("speeding", 0.0), 2.0),
            "Idling Discipline": _score_domain(v_breakdown.get("idling", 0.0), 0.2),
        }

        # Calculate fleet baseline across all vehicles
        vehicles = self.db.get_all_vehicles()
        fleet_vids = [v["vehicle_id"] for v in vehicles] if vehicles else [vehicle_id]
        fleet_domain_totals = {k: 0.0 for k in v_scores}
        fleet_dss_total = 0.0
        n_fleet = max(1, len(fleet_vids))

        for f_vid in fleet_vids:
            f_counts = counts_map.get(f_vid, {})
            f_rep = engine.calculate_score(f_counts, vehicle_id=f_vid)
            f_bd = f_rep.breakdown
            fleet_domain_totals["Fuel Integrity"] += _score_domain(f_bd.get("fuel_theft", 0.0), 2.0)
            fleet_domain_totals["Route Compliance"] += _score_domain(f_bd.get("moonlighting", 0.0), 0.5)
            fleet_domain_totals["Kinematic Smoothness"] += _score_domain(f_bd.get("harsh_driving", 0.0), 5.0)
            fleet_domain_totals["Speed Adherence"] += _score_domain(f_bd.get("speeding", 0.0), 2.0)
            fleet_domain_totals["Idling Discipline"] += _score_domain(f_bd.get("idling", 0.0), 0.2)
            fleet_dss_total += f_rep.safety_score

        fleet_averages = {k: round(v / n_fleet, 1) for k, v in fleet_domain_totals.items()}

        return {
            "vehicle_id": vehicle_id,
            "vehicle_scores": v_scores,
            "fleet_averages": fleet_averages,
            "report": vehicle_rep,
            "fleet_avg_dss": round(fleet_dss_total / n_fleet, 2),
        }

    def simulate_telegram_webhook(
        self,
        vehicle_id: str,
        incident_type: str,
        magnitude: float,
        timestamp: Optional[str] = None,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch or simulate enterprise Telegram Webhook for critical telematics alerts.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier.
        incident_type : str
            Type of critical security incident.
        magnitude : float
            Magnitude value (e.g. fuel drop volume or speed).
        timestamp : Optional[str], optional
            ISO or UTC timestamp.
        bot_token : Optional[str], optional
            Optional bot token.
        chat_id : Optional[str], optional
            Optional target Telegram chat / channel ID.

        Returns
        -------
        Dict[str, Any]
            Webhook dispatch receipt and payload.
        """
        payload_text = self.telegram_service.format_incident_payload(
            vehicle_id=vehicle_id,
            incident_type=incident_type,
            magnitude=magnitude,
            timestamp=timestamp,
        )
        receipt = self.telegram_service.dispatch_alert(
            text_message=payload_text,
            chat_id=chat_id,
            bot_token=bot_token,
        )

        # Log simulated action to audit logs
        self.db.log_admin_action(
            action="DISPATCH_TELEGRAM_WEBHOOK",
            entity_type="INCIDENT_ALERT",
            entity_id=f"{vehicle_id}_{incident_type}",
            details=f"Telegram webhook ({receipt.get('mode', 'SIMULATED')}) sent to {receipt.get('chat_id')} for {vehicle_id} ({incident_type})",
            user_id="ops_system",
        )

        return receipt

    def send_telegram_test_alert(
        self,
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch a manual verification ping to confirm webhook configuration."""
        receipt = self.telegram_service.send_test_alert(chat_id=chat_id, bot_token=bot_token)
        self.db.log_admin_action(
            action="TEST_TELEGRAM_WEBHOOK",
            entity_type="SYSTEM",
            entity_id="telegram_gateway",
            details=f"Test webhook dispatch ({receipt.get('mode', 'SIMULATED')}) to {receipt.get('chat_id')}",
            user_id="ops_system",
        )
        return receipt

    def get_replay_telemetry_snapshot(
        self,
        as_of_timestamp: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Fetch telemetry state snapshot as of a historical timeline cutoff.

        Parameters
        ----------
        as_of_timestamp : Optional[Union[str, pd.Timestamp]], optional
            Historical timeline cutoff. If None, retrieves latest state.

        Returns
        -------
        pd.DataFrame
            Vehicle telemetry records as of cutoff.
        """
        vehicles = self.db.get_all_vehicles()
        if not vehicles:
            return pd.DataFrame()

        v_df = pd.DataFrame(vehicles)

        if as_of_timestamp is not None:
            ts_cutoff_str = str(as_of_timestamp)
            query_latest = """
            SELECT p.vehicle_id, p.timestamp, p.speed, p.fuel_level, p.ignition, p.latitude, p.longitude, p.heading
            FROM telematics_pings p
            INNER JOIN (
                SELECT vehicle_id, MAX(timestamp) AS max_ts
                FROM telematics_pings
                WHERE timestamp <= ?
                GROUP BY vehicle_id
            ) latest ON p.vehicle_id = latest.vehicle_id AND p.timestamp = latest.max_ts;
            """
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query_latest, (ts_cutoff_str,))
                latest_rows = [dict(r) for r in cursor.fetchall()]
        else:
            query_latest = """
            SELECT p.vehicle_id, p.timestamp, p.speed, p.fuel_level, p.ignition, p.latitude, p.longitude, p.heading
            FROM telematics_pings p
            INNER JOIN (
                SELECT vehicle_id, MAX(timestamp) AS max_ts
                FROM telematics_pings
                GROUP BY vehicle_id
            ) latest ON p.vehicle_id = latest.vehicle_id AND p.timestamp = latest.max_ts;
            """
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query_latest)
                latest_rows = [dict(r) for r in cursor.fetchall()]

        latest_df = pd.DataFrame(latest_rows) if latest_rows else pd.DataFrame()

        if not latest_df.empty:
            merged = pd.merge(v_df, latest_df, on="vehicle_id", how="left")
        else:
            merged = v_df.copy()
            for col in ["timestamp", "speed", "fuel_level", "ignition", "latitude", "longitude", "heading"]:
                merged[col] = None

        def format_ignition(val: Any) -> str:
            if pd.isna(val):
                return "Unknown"
            if str(val) in ("1", "ON", "True", "1.0"):
                return "ON"
            return "OFF"

        merged["ignition_label"] = merged["ignition"].apply(format_ignition)
        return merged


@st.cache_resource
def get_data_service(db_path: Optional[str] = None) -> TelematicsDataService:
    """Streamlit cached singleton factory for TelematicsDataService."""
    return TelematicsDataService(db_path=db_path)
