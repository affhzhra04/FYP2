"""SQLite Database and CRUD Integration Manager for FYP2 Telematics Core.

This module provides transaction-safe, production-grade SQLite storage management
for commercial IoT telematics fleet profiles, driver schedules (night-shift vs. day-shift),
and anomaly event logs.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Sequence

import pandas as pd

# Module logger configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default persistent database location within repository
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "telematics_core.db"


class DatabaseManagerError(Exception):
    """Base exception for telematics database operations."""
    pass


class TelematicsDBManager:
    """Manages SQLite persistence, schema migrations, and CRUD operations for Telematics Core.

    Attributes:
        db_path (Path): Path to the SQLite database file.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
        auto_seed: bool = True,
    ) -> None:
        """Initialize the database manager.

        Parameters
        ----------
        db_path : Optional[Union[str, Path]], optional
            Path to SQLite database file. Defaults to 'database/telematics_core.db'.
            Pass ':memory:' for in-memory testing.
        auto_init : bool, optional
            Automatically run schema initialization if True. Default is True.
        auto_seed : bool, optional
            Automatically seed baseline mock data if tables are empty. Default is True.
        """
        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
            self._is_memory = False
            self._keep_alive = None
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
            self._is_memory = True
            self._mem_uri = f"file:memdb_{id(self)}?mode=memory&cache=shared"
            self._keep_alive = sqlite3.connect(self._mem_uri, uri=True)
            self._keep_alive.execute("PRAGMA foreign_keys = ON;")
        else:
            self.db_path = Path(db_path).resolve()
            self._is_memory = False
            self._keep_alive = None

        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("TelematicsDBManager initialized at: %s", self.db_path)

        if auto_init:
            self.init_db()
            if auto_seed:
                self.seed_data()

    def __del__(self) -> None:
        """Close keep-alive connection if active."""
        if getattr(self, "_keep_alive", None) is not None:
            try:
                self._keep_alive.close()
            except Exception:
                pass

    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional SQLite connection with foreign keys and Row factory enabled.

        Yields
        ------
        sqlite3.Connection
            Active SQLite connection.

        Raises
        ------
        DatabaseManagerError
            If connection fails or an unhandled SQL error occurs.
        """
        conn_target = self._mem_uri if self._is_memory else str(self.db_path)
        try:
            if self._is_memory:
                conn = sqlite3.connect(conn_target, uri=True)
            else:
                conn = sqlite3.connect(conn_target)
            conn.row_factory = sqlite3.Row
            # Enforce foreign key constraints in SQLite
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
        except sqlite3.Error as exc:
            logger.error("Failed to connect to SQLite database at %s: %s", conn_target, exc)
            raise DatabaseManagerError(f"Database connection error: {exc}") from exc

        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("Transaction rolled back due to error: %s", exc)
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize SQLite database tables and performance indexes if they do not exist.

        Tables Created:
            - vehicles: Master registry of commercial telematics vehicles.
            - driver_schedules: Shift rosters with is_night_shift flags.
            - anomaly_logs: Incident telemetry events and tracking status.
        """
        create_vehicles_table_sql = """
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id TEXT PRIMARY KEY,
            plate_number TEXT NOT NULL,
            model TEXT NOT NULL,
            fuel_capacity_liters REAL NOT NULL
        );
        """

        create_driver_schedules_table_sql = """
        CREATE TABLE IF NOT EXISTS driver_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id TEXT NOT NULL,
            vehicle_id TEXT NOT NULL,
            shift_start TEXT NOT NULL,
            shift_end TEXT NOT NULL,
            is_night_shift INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE
        );
        """

        create_anomaly_logs_table_sql = """
        CREATE TABLE IF NOT EXISTS anomaly_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            magnitude REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            FOREIGN KEY(vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE
        );
        """

        create_telematics_pings_table_sql = """
        CREATE TABLE IF NOT EXISTS telematics_pings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            speed REAL NOT NULL,
            fuel_level REAL NOT NULL,
            heading REAL,
            ignition INTEGER,
            delta_time REAL,
            delta_speed REAL,
            delta_fuel REAL,
            FOREIGN KEY(vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE
        );
        """

        create_admin_audit_logs_table_sql = """
        CREATE TABLE IF NOT EXISTS admin_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            details TEXT,
            user_id TEXT NOT NULL DEFAULT 'admin'
        );
        """

        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_driver_schedules_vehicle ON driver_schedules(vehicle_id);",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_logs_vehicle ON anomaly_logs(vehicle_id);",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_logs_status ON anomaly_logs(status);",
            "CREATE INDEX IF NOT EXISTS idx_telematics_pings_vehicle_ts ON telematics_pings(vehicle_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_admin_audit_logs_ts ON admin_audit_logs(timestamp);",
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_vehicles_table_sql)
            cursor.execute(create_driver_schedules_table_sql)
            cursor.execute(create_anomaly_logs_table_sql)
            cursor.execute(create_telematics_pings_table_sql)
            cursor.execute(create_admin_audit_logs_table_sql)
            for idx_stmt in create_indexes_sql:
                cursor.execute(idx_stmt)

        logger.info("Telematics schema initialized successfully.")

    def seed_data(self) -> bool:
        """Seed baseline mock vehicles and driver shift schedules if the database is empty.

        Seeds:
            - Vehicle '8_9_8' (Volvo FH16) with designated Night Shift (22:00 - 06:00, is_night_shift=1).
            - Vehicle '4_2_1' (Scania R500) with Standard Day Shift (08:00 - 17:00, is_night_shift=0).

        Returns
        -------
        bool
            True if sample data was newly seeded, False if tables already had records.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM vehicles;")
            row = cursor.fetchone()
            count = row["cnt"] if row else 0

            if count > 0:
                logger.info("Database already contains %d vehicle records. Skipping seeding.", count)
                return False

            logger.info("Seeding initial mock vehicles and driver schedules...")

            # 1. Seed sample commercial vehicles
            sample_vehicles = [
                ("8_9_8", "WXY-8988", "Volvo FH16 Commercial Heavy Hauler", 450.0),
                ("4_2_1", "VBC-4210", "Scania R500 Long-Haul Logistics Truck", 380.0),
            ]
            cursor.executemany(
                """
                INSERT OR IGNORE INTO vehicles (vehicle_id, plate_number, model, fuel_capacity_liters)
                VALUES (?, ?, ?, ?);
                """,
                sample_vehicles,
            )

            # 2. Seed driver shift schedules (including designated night shift for dynamic threshold testing)
            sample_schedules = [
                ("DRV_NIGHT_898", "8_9_8", "22:00:00", "06:00:00", 1),  # Night Shift: Active
                ("DRV_DAY_421", "4_2_1", "08:00:00", "17:00:00", 0),    # Day Shift: Standard
            ]
            cursor.executemany(
                """
                INSERT INTO driver_schedules (driver_id, vehicle_id, shift_start, shift_end, is_night_shift)
                VALUES (?, ?, ?, ?, ?);
                """,
                sample_schedules,
            )

        logger.info("Seeding completed successfully.")
        return True

    # --------------------------------------------------------------------------
    # CRUD Operations: Create / Insert
    # --------------------------------------------------------------------------

    def add_vehicle(
        self,
        vehicle_id: str,
        plate_number: str,
        model: str,
        fuel_capacity_liters: float,
    ) -> bool:
        """Insert a new commercial vehicle record into the fleet registry.

        Parameters
        ----------
        vehicle_id : str
            Unique vehicle identifier (e.g. '8_9_8').
        plate_number : str
            Official vehicle license plate.
        model : str
            Vehicle make, model, or category.
        fuel_capacity_liters : float
            Maximum tank fuel capacity in liters.

        Returns
        -------
        bool
            True if vehicle was inserted, False if a duplicate vehicle_id exists.
        """
        query = """
        INSERT INTO vehicles (vehicle_id, plate_number, model, fuel_capacity_liters)
        VALUES (?, ?, ?, ?);
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    query,
                    (str(vehicle_id), str(plate_number), str(model), float(fuel_capacity_liters)),
                )
            logger.info("Added vehicle: %s (%s)", vehicle_id, plate_number)
            return True
        except sqlite3.IntegrityError:
            logger.warning("Vehicle %s already exists. Insertion aborted.", vehicle_id)
            return False

    def add_driver_schedule(
        self,
        driver_id: str,
        vehicle_id: str,
        shift_start: str,
        shift_end: str,
        is_night_shift: Union[bool, int],
    ) -> int:
        """Insert a driver shift schedule associated with a commercial vehicle.

        Parameters
        ----------
        driver_id : str
            Driver identification code.
        vehicle_id : str
            Vehicle assigned to this shift.
        shift_start : str
            Shift start time (e.g. '22:00:00').
        shift_end : str
            Shift end time (e.g. '06:00:00').
        is_night_shift : Union[bool, int]
            True/1 if this is a night-shift roster requiring dynamic anomaly thresholds.

        Returns
        -------
        int
            The autoincremented primary key ID of the inserted schedule.

        Raises
        ------
        ValueError
            If foreign key violation occurs (vehicle does not exist).
        """
        query = """
        INSERT INTO driver_schedules (driver_id, vehicle_id, shift_start, shift_end, is_night_shift)
        VALUES (?, ?, ?, ?, ?);
        """
        night_shift_flag = 1 if is_night_shift else 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (str(driver_id), str(vehicle_id), str(shift_start), str(shift_end), night_shift_flag),
                )
                schedule_id = cursor.lastrowid
            logger.info(
                "Inserted driver schedule #%d for driver %s on vehicle %s (Night Shift: %s)",
                schedule_id,
                driver_id,
                vehicle_id,
                bool(night_shift_flag),
            )
            return schedule_id
        except sqlite3.IntegrityError as exc:
            logger.error("Foreign key or constraint error inserting schedule for vehicle %s: %s", vehicle_id, exc)
            raise ValueError(f"Vehicle '{vehicle_id}' does not exist in registry.") from exc

    def log_anomaly(
        self,
        vehicle_id: str,
        timestamp: str,
        incident_type: str,
        magnitude: float,
        status: str = "OPEN",
    ) -> int:
        """Log a detected telematics anomaly incident (e.g. sudden fuel drop, severe speeding).

        Parameters
        ----------
        vehicle_id : str
            Vehicle that generated the anomaly.
        timestamp : str
            ISO UTC timestamp of the detected incident.
        incident_type : str
            Categorization (e.g. 'SUDDEN_FUEL_DROP', 'OVERSPEED_SPIKE', 'HARSH_BRAKING').
        magnitude : float
            Quantitative magnitude of the anomaly (e.g. -15.5 Liters, 145.0 km/h).
        status : str, optional
            Lifecycle status of incident ('OPEN', 'RESOLVED', etc.). Defaults to 'OPEN'.

        Returns
        -------
        int
            The autoincremented primary key ID of the logged anomaly incident.
        """
        query = """
        INSERT INTO anomaly_logs (vehicle_id, timestamp, incident_type, magnitude, status)
        VALUES (?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (str(vehicle_id), str(timestamp), str(incident_type), float(magnitude), str(status).upper()),
            )
            anomaly_id = cursor.lastrowid

        logger.info(
            "Logged anomaly #%d for vehicle %s [%s, magnitude=%.2f, status=%s]",
            anomaly_id,
            vehicle_id,
            incident_type,
            magnitude,
            status.upper(),
        )
        return anomaly_id

    # --------------------------------------------------------------------------
    # CRUD Operations: Read / Query
    # --------------------------------------------------------------------------

    def get_vehicle(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single vehicle profile by vehicle ID.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier to query.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary of vehicle details, or None if not found.
        """
        query = """
        SELECT vehicle_id, plate_number, model, fuel_capacity_liters
        FROM vehicles
        WHERE vehicle_id = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(vehicle_id),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """Fetch all registered vehicle profiles in the fleet.

        Returns
        -------
        List[Dict[str, Any]]
            List of vehicle records.
        """
        query = """
        SELECT vehicle_id, plate_number, model, fuel_capacity_liters
        FROM vehicles
        ORDER BY vehicle_id ASC;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_driver_schedules(self, vehicle_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve driver shift schedules, optionally filtered by vehicle ID.

        Parameters
        ----------
        vehicle_id : Optional[str], optional
            If specified, filter schedules for this vehicle only.

        Returns
        -------
        List[Dict[str, Any]]
            List of schedule records with is_night_shift converted to boolean.
        """
        if vehicle_id is not None:
            query = """
            SELECT id, driver_id, vehicle_id, shift_start, shift_end, is_night_shift
            FROM driver_schedules
            WHERE vehicle_id = ?
            ORDER BY id ASC;
            """
            params = (str(vehicle_id),)
        else:
            query = """
            SELECT id, driver_id, vehicle_id, shift_start, shift_end, is_night_shift
            FROM driver_schedules
            ORDER BY id ASC;
            """
            params = ()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            record = dict(r)
            record["is_night_shift"] = bool(record["is_night_shift"])
            results.append(record)
        return results

    def is_night_shift(self, vehicle_id: str) -> bool:
        """Check if any designated night shift schedule is active for a given vehicle ID.

        Used by anomaly detection engines to dynamically relax or tighten thresholds
        during night-time telemetry intervals.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier to inspect.

        Returns
        -------
        bool
            True if the vehicle has an active night-shift schedule profile, False otherwise.
        """
        query = """
        SELECT COUNT(*) AS night_count
        FROM driver_schedules
        WHERE vehicle_id = ? AND is_night_shift = 1;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(vehicle_id),))
            row = cursor.fetchone()
            count = row["night_count"] if row else 0
            return count > 0

    def get_anomaly_logs(
        self,
        vehicle_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch telematics anomaly logs with optional filtering by vehicle ID and status.

        Parameters
        ----------
        vehicle_id : Optional[str], optional
            Filter incidents by vehicle ID.
        status : Optional[str], optional
            Filter incidents by lifecycle status (e.g. 'OPEN', 'RESOLVED').

        Returns
        -------
        List[Dict[str, Any]]
            List of anomaly incident records sorted chronologically.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if vehicle_id is not None:
            clauses.append("vehicle_id = ?")
            params.append(str(vehicle_id))
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status).upper())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
        SELECT id, vehicle_id, timestamp, incident_type, magnitude, status
        FROM anomaly_logs
        {where_sql}
        ORDER BY id DESC;
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def get_anomaly_by_id(self, anomaly_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single anomaly log record by primary key ID.

        Parameters
        ----------
        anomaly_id : int
            Primary key identifier of anomaly record.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary of anomaly incident details, or None if not found.
        """
        query = """
        SELECT id, vehicle_id, timestamp, incident_type, magnitude, status
        FROM anomaly_logs
        WHERE id = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (int(anomaly_id),))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --------------------------------------------------------------------------
    # CRUD Operations: Update
    # --------------------------------------------------------------------------

    def update_anomaly_status(self, anomaly_id: int, new_status: str) -> bool:
        """Update an anomaly incident log's status.

        Parameters
        ----------
        anomaly_id : int
            Primary key identifier of anomaly record.
        new_status : str
            New status to set (e.g. 'RESOLVED', 'OPEN', 'UNDER_INVESTIGATION').

        Returns
        -------
        bool
            True if a row was updated, False if the record ID does not exist.
        """
        query = """
        UPDATE anomaly_logs
        SET status = ?
        WHERE id = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(new_status).upper(), int(anomaly_id)))
            updated_count = cursor.rowcount

        success = updated_count > 0
        if success:
            logger.info("Updated anomaly #%d status to %s", anomaly_id, new_status.upper())
        else:
            logger.warning("Anomaly #%d not found for status update.", anomaly_id)
        return success

    def resolve_anomaly(self, anomaly_id: int) -> bool:
        """Convenience method to transition an anomaly log's status from 'OPEN' to 'RESOLVED'.

        Parameters
        ----------
        anomaly_id : int
            Primary key identifier of anomaly record.

        Returns
        -------
        bool
            True if status was transitioned to 'RESOLVED', False otherwise.
        """
        return self.update_anomaly_status(anomaly_id=anomaly_id, new_status="RESOLVED")

    # --------------------------------------------------------------------------
    # CRUD Operations: Delete
    # --------------------------------------------------------------------------

    def delete_vehicle(self, vehicle_id: str) -> bool:
        """Delete a vehicle record from the registry and cascade deletes to related records.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier to remove.

        Returns
        -------
        bool
            True if a vehicle record was deleted, False if not found.
        """
        query = "DELETE FROM vehicles WHERE vehicle_id = ?;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(vehicle_id),))
            deleted_count = cursor.rowcount

        success = deleted_count > 0
        if success:
            logger.info("Deleted vehicle: %s (and cascaded associations)", vehicle_id)
        else:
            logger.warning("Vehicle %s not found for deletion.", vehicle_id)
        return success

    def delete_anomaly_log(self, anomaly_id: int) -> bool:
        """Delete an anomaly incident log by primary key ID.

        Parameters
        ----------
        anomaly_id : int
            Primary key identifier of anomaly record.

        Returns
        -------
        bool
            True if the anomaly was deleted, False if record does not exist.
        """
        query = "DELETE FROM anomaly_logs WHERE id = ?;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (int(anomaly_id),))
            deleted_count = cursor.rowcount

        success = deleted_count > 0
        if success:
            logger.info("Deleted anomaly log #%d", anomaly_id)
        else:
            logger.warning("Anomaly log #%d not found for deletion.", anomaly_id)
        return success

    def delete_driver_schedule(self, schedule_id: int) -> bool:
        """Delete a driver shift schedule by primary key ID.

        Parameters
        ----------
        schedule_id : int
            Primary key identifier of driver schedule record.

        Returns
        -------
        bool
            True if schedule was deleted, False if not found.
        """
        query = "DELETE FROM driver_schedules WHERE id = ?;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (int(schedule_id),))
            deleted_count = cursor.rowcount

        success = deleted_count > 0
        if success:
            logger.info("Deleted driver schedule #%d", schedule_id)
        else:
            logger.warning("Driver schedule #%d not found for deletion.", schedule_id)
        return success

    # --------------------------------------------------------------------------
    # Telematics Pings Storage & Retrieval
    # --------------------------------------------------------------------------

    def save_telematics_pings(
        self,
        df: Any,
        batch_size: int = 10000,
    ) -> int:
        """Save cleaned telematics pings into telematics_pings table in high-performance batches.

        Automatically ensures all associated vehicle_id values exist in the vehicles table.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned telematics DataFrame containing vehicle_id, timestamp, latitude,
            longitude, speed, fuel_level, and optional delta features.
        batch_size : int, optional
            Number of records per executemany chunk. Defaults to 10000.

        Returns
        -------
        int
            Number of records successfully inserted.
        """
        if df is None or len(df) == 0:
            return 0

        # Auto-register any new vehicles
        unique_vids = df["vehicle_id"].dropna().unique()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            vehicle_tuples = [
                (str(vid), f"PLT-{vid}", "Innonics Commercial Fleet", 450.0)
                for vid in unique_vids
            ]
            cursor.executemany(
                """
                INSERT OR IGNORE INTO vehicles (vehicle_id, plate_number, model, fuel_capacity_liters)
                VALUES (?, ?, ?, ?);
                """,
                vehicle_tuples,
            )

        # Prepare records for insertion
        insert_sql = """
        INSERT INTO telematics_pings (
            vehicle_id, timestamp, latitude, longitude, speed, fuel_level,
            heading, ignition, delta_time, delta_speed, delta_fuel
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        has_heading = "heading" in df.columns
        has_ignition = "ignition" in df.columns
        has_dt = "delta_time" in df.columns
        has_dv = "delta_speed" in df.columns
        has_df = "delta_fuel" in df.columns

        records: List[Tuple[Any, ...]] = []
        for row in df.itertuples(index=False):
            # Extract standard attributes
            vid = getattr(row, "vehicle_id")
            ts = str(getattr(row, "timestamp"))
            lat = float(getattr(row, "latitude"))
            lon = float(getattr(row, "longitude"))
            spd = float(getattr(row, "speed"))
            fuel = float(getattr(row, "fuel_level"))

            heading = float(getattr(row, "heading")) if has_heading and not pd.isna(getattr(row, "heading")) else None
            ign = int(getattr(row, "ignition")) if has_ignition and not pd.isna(getattr(row, "ignition")) else None
            dt = float(getattr(row, "delta_time")) if has_dt and not pd.isna(getattr(row, "delta_time")) else None
            dv = float(getattr(row, "delta_speed")) if has_dv and not pd.isna(getattr(row, "delta_speed")) else None
            dl = float(getattr(row, "delta_fuel")) if has_df and not pd.isna(getattr(row, "delta_fuel")) else None

            records.append((vid, ts, lat, lon, spd, fuel, heading, ign, dt, dv, dl))

        total_inserted = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for i in range(0, len(records), batch_size):
                chunk = records[i : i + batch_size]
                cursor.executemany(insert_sql, chunk)
                total_inserted += len(chunk)

        logger.info("Persisted %d telematics pings into database.", total_inserted)
        return total_inserted

    def get_telematics_pings(
        self,
        vehicle_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical telematics pings from telematics_pings table.

        Parameters
        ----------
        vehicle_id : Optional[str], optional
            Filter by specific vehicle ID.
        limit : Optional[int], optional
            Maximum records to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of ping dictionaries.
        """
        clauses = []
        params = []
        if vehicle_id is not None:
            clauses.append("vehicle_id = ?")
            params.append(str(vehicle_id))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

        query = f"""
        SELECT id, vehicle_id, timestamp, latitude, longitude, speed, fuel_level,
               heading, ignition, delta_time, delta_speed, delta_fuel
        FROM telematics_pings
        {where_sql}
        ORDER BY vehicle_id ASC, timestamp ASC
        {limit_sql};
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def log_anomalies_bulk(
        self,
        anomalies: Sequence[Any],
        batch_size: int = 5000,
    ) -> int:
        """Bulk insert detected AnomalyEvent instances into anomaly_logs table.

        Parameters
        ----------
        anomalies : Sequence[Any]
            Collection of AnomalyEvent instances or dictionaries.
        batch_size : int, optional
            Batch size for bulk execution. Defaults to 5000.

        Returns
        -------
        int
            Number of logged anomalies.
        """
        if not anomalies:
            return 0

        insert_sql = """
        INSERT INTO anomaly_logs (vehicle_id, timestamp, incident_type, magnitude, status)
        VALUES (?, ?, ?, ?, ?);
        """

        records = []
        for anom in anomalies:
            if hasattr(anom, "vehicle_id"):
                vid = str(anom.vehicle_id)
                ts = str(anom.timestamp)
                i_type = str(anom.incident_type)
                mag = float(anom.magnitude)
                status = str(anom.status).upper()
            elif isinstance(anom, dict):
                vid = str(anom.get("vehicle_id", ""))
                ts = str(anom.get("timestamp", ""))
                i_type = str(anom.get("incident_type", ""))
                mag = float(anom.get("magnitude", 0.0))
                status = str(anom.get("status", "OPEN")).upper()
            else:
                continue
            records.append((vid, ts, i_type, mag, status))

        total_logged = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for i in range(0, len(records), batch_size):
                chunk = records[i : i + batch_size]
                cursor.executemany(insert_sql, chunk)
                total_logged += len(chunk)

        logger.info("Bulk-logged %d anomaly incidents into database.", total_logged)
        return total_logged

    # --------------------------------------------------------------------------
    # Administrative CRUD Updates & Audit Logging
    # --------------------------------------------------------------------------

    def update_vehicle(
        self,
        vehicle_id: str,
        plate_number: str,
        model: str,
        fuel_capacity_liters: float,
    ) -> bool:
        """Update existing vehicle registration attributes.

        Parameters
        ----------
        vehicle_id : str
            Vehicle identifier to update.
        plate_number : str
            Updated license plate number.
        model : str
            Updated vehicle model/make.
        fuel_capacity_liters : float
            Updated fuel capacity in Liters.

        Returns
        -------
        bool
            True if vehicle was updated, False if not found.
        """
        query = """
        UPDATE vehicles
        SET plate_number = ?, model = ?, fuel_capacity_liters = ?
        WHERE vehicle_id = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (str(plate_number), str(model), float(fuel_capacity_liters), str(vehicle_id)),
            )
            updated = cursor.rowcount > 0

        if updated:
            self.log_admin_action(
                action="UPDATE",
                entity_type="VEHICLE",
                entity_id=vehicle_id,
                details=f"Plate: {plate_number}, Model: {model}, Capacity: {fuel_capacity_liters}L",
            )
        return updated

    def update_driver_schedule(
        self,
        schedule_id: int,
        driver_id: str,
        vehicle_id: str,
        shift_start: str,
        shift_end: str,
        is_night_shift: Union[bool, int],
    ) -> bool:
        """Update an existing driver shift schedule.

        Parameters
        ----------
        schedule_id : int
            Schedule record primary key.
        driver_id : str
            Driver identification code.
        vehicle_id : str
            Vehicle assigned to this roster.
        shift_start : str
            Shift start time (e.g. '22:00:00').
        shift_end : str
            Shift end time (e.g. '06:00:00').
        is_night_shift : Union[bool, int]
            Flag indicating night-shift operational window.

        Returns
        -------
        bool
            True if schedule was updated, False if not found.
        """
        query = """
        UPDATE driver_schedules
        SET driver_id = ?, vehicle_id = ?, shift_start = ?, shift_end = ?, is_night_shift = ?
        WHERE id = ?;
        """
        flag = 1 if is_night_shift else 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (str(driver_id), str(vehicle_id), str(shift_start), str(shift_end), flag, int(schedule_id)),
            )
            updated = cursor.rowcount > 0

        if updated:
            self.log_admin_action(
                action="UPDATE",
                entity_type="DRIVER_SCHEDULE",
                entity_id=str(schedule_id),
                details=f"Driver: {driver_id}, Vehicle: {vehicle_id}, Shift: {shift_start}-{shift_end}, NightShift: {bool(flag)}",
            )
        return updated

    def update_anomaly_with_resolution(
        self,
        anomaly_id: int,
        status: str,
        resolution_notes: str,
        user_id: str = "admin",
    ) -> bool:
        """Update an anomaly incident status and record administrative audit rationale.

        Parameters
        ----------
        anomaly_id : int
            Anomaly incident primary key.
        status : str
            New incident status ('RESOLVED', 'OPEN', 'UNDER_INVESTIGATION').
        resolution_notes : str
            Rationale/operational notes for the status change.
        user_id : str, optional
            Authorized administrative user ID. Defaults to 'admin'.

        Returns
        -------
        bool
            True if anomaly was updated, False otherwise.
        """
        success = self.update_anomaly_status(anomaly_id=anomaly_id, new_status=status)
        if success:
            self.log_admin_action(
                action="RESOLVE_ANOMALY" if status.upper() == "RESOLVED" else "UPDATE_ANOMALY_STATUS",
                entity_type="ANOMALY",
                entity_id=str(anomaly_id),
                details=f"New Status: {status.upper()} | Reason: {resolution_notes}",
                user_id=user_id,
            )
        return success

    def log_admin_action(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        details: str,
        user_id: str = "admin",
    ) -> int:
        """Record an administrative change action into the audit trail.

        Parameters
        ----------
        action : str
            Action type (e.g. 'CREATE', 'UPDATE', 'DELETE', 'RESOLVE').
        entity_type : str
            Target entity (e.g. 'VEHICLE', 'DRIVER_SCHEDULE', 'ANOMALY').
        entity_id : str
            Entity primary key or identifier.
        details : str
            Audit notes / changelog description.
        user_id : str, optional
            Administrative actor username. Defaults to 'admin'.

        Returns
        -------
        int
            Audit record primary key ID.
        """
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        query = """
        INSERT INTO admin_audit_logs (timestamp, action, entity_type, entity_id, details, user_id)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (now_utc, str(action).upper(), str(entity_type).upper(), str(entity_id), str(details), str(user_id)),
            )
            audit_id = cursor.lastrowid
        return audit_id

    def get_admin_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent administrative audit log entries.

        Parameters
        ----------
        limit : int, optional
            Maximum records to return. Defaults to 100.

        Returns
        -------
        List[Dict[str, Any]]
            Chronological audit log records (most recent first).
        """
        query = """
        SELECT id, timestamp, action, entity_type, entity_id, details, user_id
        FROM admin_audit_logs
        ORDER BY id DESC
        LIMIT ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (int(limit),))
            return [dict(r) for r in cursor.fetchall()]
