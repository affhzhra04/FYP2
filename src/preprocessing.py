"""Production-grade Telematics Preprocessing and Data Quality Engine.

This module provides high-throughput, vectorized ingestion, data quality validation,
sanitization, and trajectory feature precomputations for commercial IoT telematics data.
Engineered for scale, handling 300,000+ records in sub-second execution windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class TelematicsAuditMetrics:
    """Canonical schema for Data Quality Audit Report."""

    initial_records: int
    valid_records: int
    dropped_invalid_coordinates: int
    dropped_invalid_speed: int
    dropped_invalid_fuel: int
    dropped_duplicate_timestamps: int
    data_retention_rate_pct: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Serialize audit metrics to standard dictionary."""
        return {
            "initial_records": self.initial_records,
            "valid_records": self.valid_records,
            "dropped_invalid_coordinates": self.dropped_invalid_coordinates,
            "dropped_invalid_speed": self.dropped_invalid_speed,
            "dropped_invalid_fuel": self.dropped_invalid_fuel,
            "dropped_duplicate_timestamps": self.dropped_duplicate_timestamps,
            "data_retention_rate_pct": self.data_retention_rate_pct,
        }


class TelematicsCleaner:
    """Production-grade cleaner and feature engineering precomputer for IoT telematics.

    Applies vectorized NumPy/Pandas filtering for spatial-temporal validation,
    sensor sanity boundaries, ping deduplication, and kinematics feature engineering
    (time interval ΔT, velocity differential ΔV, and fuel rate ΔL).

    Attributes:
        min_lat: Minimum allowable latitude in degrees (default: -90.0).
        max_lat: Maximum allowable latitude in degrees (default: 90.0).
        min_lon: Minimum allowable longitude in degrees (default: -180.0).
        max_lon: Maximum allowable longitude in degrees (default: 180.0).
        min_speed: Minimum physically feasible speed in km/h (default: 0.0).
        max_speed: Maximum physically feasible speed in km/h (default: 160.0).
        sentinel_fuel_values: Disconnected sensor sentinels (default: {-999.0, -9999.0, -1.0}).
        vehicle_id_col: Column name identifying the vehicle / asset.
        timestamp_col: Column name containing datetime/timestamp ping.
        latitude_col: Column name for latitude.
        longitude_col: Column name for longitude.
        speed_col: Column name for speed in km/h.
        fuel_col: Column name for fuel level in liters.
        drop_invalid_speed: Whether to drop rows violating speed boundaries.
    """

    DEFAULT_SENTINEL_FUEL_VALUES: Tuple[float, ...] = (-999.0, -9999.0, -1.0)

    # Standard column alias mappings for plug-and-play IoT compatibility
    COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
        "vehicle_id": ("vehicle_id", "vehicle", "device_id", "vid", "asset_id", "imei"),
        "timestamp": ("timestamp", "datetime", "time", "ts", "ping_time", "event_time"),
        "latitude": ("latitude", "lat", "y"),
        "longitude": ("longitude", "lon", "lng", "long", "x"),
        "speed": ("speed", "speed_kmh", "velocity", "spd", "speed_kph"),
        "fuel_level": ("fuel_level", "fuel", "fuel_liters", "fuel_l", "fuel_volume", "fuel_reading"),
    }

    def __init__(
        self,
        min_lat: float = -90.0,
        max_lat: float = 90.0,
        min_lon: float = -180.0,
        max_lon: float = 180.0,
        min_speed: float = 0.0,
        max_speed: float = 160.0,
        sentinel_fuel_values: Optional[Sequence[float]] = None,
        vehicle_id_col: str = "vehicle_id",
        timestamp_col: str = "timestamp",
        latitude_col: str = "latitude",
        longitude_col: str = "longitude",
        speed_col: str = "speed",
        fuel_col: str = "fuel_level",
        delta_time_col: str = "delta_time",
        delta_speed_col: str = "delta_speed",
        delta_fuel_col: str = "delta_fuel",
        drop_invalid_speed: bool = True,
        strict_columns: bool = False,
        target_timezone: Optional[str] = None,
    ) -> None:
        """Initialize TelematicsCleaner with boundary thresholds and schema mapping."""
        self.min_lat = float(min_lat)
        self.max_lat = float(max_lat)
        self.min_lon = float(min_lon)
        self.max_lon = float(max_lon)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.target_timezone = target_timezone

        if sentinel_fuel_values is None:
            self.sentinel_fuel_values: Set[float] = set(self.DEFAULT_SENTINEL_FUEL_VALUES)
        else:
            self.sentinel_fuel_values = {float(v) for v in sentinel_fuel_values}

        self.vehicle_id_col = vehicle_id_col
        self.timestamp_col = timestamp_col
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col
        self.speed_col = speed_col
        self.fuel_col = fuel_col

        self.delta_time_col = delta_time_col
        self.delta_speed_col = delta_speed_col
        self.delta_fuel_col = delta_fuel_col

        self.drop_invalid_speed = drop_invalid_speed
        self.strict_columns = strict_columns

    def ingest(
        self,
        data: Union[pd.DataFrame, Sequence[Dict[str, Any]], str, Path],
    ) -> pd.DataFrame:
        """Ingest commercial telematics data from DataFrames, JSON records, or file paths.

        Parameters
        ----------
        data : Union[pd.DataFrame, Sequence[Dict[str, Any]], str, Path]
            Data source to ingest. Can be a pandas DataFrame, a list/sequence of dicts,
            or a path/JSON string.

        Returns
        -------
        pd.DataFrame
            Normalized pandas DataFrame ready for cleaning.

        Raises
        ------
        TypeError
            If input data type is unsupported.
        ValueError
            If input string/path cannot be parsed into a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, (list, tuple)):
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(list(data))

        if isinstance(data, (str, Path)):
            path_obj = Path(data)
            if path_obj.exists() and path_obj.is_file():
                if path_obj.suffix.lower() == ".parquet":
                    return pd.read_parquet(path_obj)
                if path_obj.suffix.lower() in (".json", ".jsonl"):
                    try:
                        return pd.read_json(path_obj, lines=True)
                    except ValueError:
                        return pd.read_json(path_obj)
                if path_obj.suffix.lower() == ".csv":
                    return pd.read_csv(path_obj)
            # Try reading as raw JSON string
            try:
                import io
                return pd.read_json(io.StringIO(str(data)))
            except Exception as exc:
                raise ValueError(f"Unable to parse input string/path as tabular data: {exc}") from exc

        raise TypeError(
            f"Unsupported data type for telematics ingestion: {type(data).__name__}. "
            "Expected pd.DataFrame, Sequence[Dict[str, Any]], or file path."
        )

    def _resolve_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """Resolve expected column names against DataFrame columns, checking aliases.

        Returns a dictionary mapping canonical role -> actual column name in df.
        """
        resolved: Dict[str, str] = {}
        target_roles = {
            "vehicle_id": self.vehicle_id_col,
            "timestamp": self.timestamp_col,
            "latitude": self.latitude_col,
            "longitude": self.longitude_col,
            "speed": self.speed_col,
            "fuel_level": self.fuel_col,
        }

        existing_cols_lower = {col.lower(): col for col in df.columns}

        for role, configured_name in target_roles.items():
            if configured_name in df.columns:
                resolved[role] = configured_name
                continue

            if not self.strict_columns:
                # Check case-insensitive match
                if configured_name.lower() in existing_cols_lower:
                    resolved[role] = existing_cols_lower[configured_name.lower()]
                    continue

                # Check alias mapping
                matched_alias = None
                for alias in self.COLUMN_ALIASES.get(role, ()):
                    if alias in df.columns:
                        matched_alias = alias
                        break
                    if alias.lower() in existing_cols_lower:
                        matched_alias = existing_cols_lower[alias.lower()]
                        break

                if matched_alias:
                    resolved[role] = matched_alias
                    continue

            # Fall back to configured name (will be flagged if missing)
            resolved[role] = configured_name

        return resolved

    def _validate_input_schema(self, df: pd.DataFrame, cols: Dict[str, str]) -> None:
        """Validate that all essential telematics fields are present in the DataFrame."""
        missing = [name for role, name in cols.items() if name not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required telematics column(s): {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def clean_data(
        self,
        df: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    ) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
        """Execute the production telematics cleaning, quality audit, and feature precomputations.

        Pipeline stages:
          1. Coordinate Validity: Drop rows with NaN or out-of-range latitude/longitude.
          2. Speed Sanitization: Drop or flag records with speed < 0 km/h or > 160 km/h.
          3. Fuel Reading Integrity: Drop disconnected sensor values (< 0 or sentinels like -999).
          4. Timestamp Parsing & Deduplication: Convert to standard UTC datetime64[ns],
             sort by (vehicle_id, timestamp), and remove exact duplicate pings for the vehicle.
          5. Feature Engineering Precomputations: Vectorized computation of:
             - ΔT = T_n - T_{n-1} (seconds)
             - ΔV = V_n - V_{n-1} (km/h)
             - ΔL = L_n - L_{n-1} (Liters)
             grouped by vehicle_id. Drop records where ΔT <= 0.
          6. Data Quality Audit: Return cleaned DataFrame with comprehensive audit dictionary.

        Parameters
        ----------
        df : Union[pd.DataFrame, Sequence[Dict[str, Any]]]
            Commercial IoT telematics data as DataFrame or list of JSON records.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Union[int, float]]]
            - Cleaned and feature-engineered DataFrame.
            - Audit report dictionary with exact metric keys:
                - initial_records
                - valid_records
                - dropped_invalid_coordinates
                - dropped_invalid_speed
                - dropped_invalid_fuel
                - dropped_duplicate_timestamps
                - data_retention_rate_pct
        """
        # Ingestion normalization
        if not isinstance(df, pd.DataFrame):
            df = self.ingest(df)
        else:
            df = df.copy()

        initial_records = len(df)
        logger.info("Initiating TelematicsCleaner pipeline on %d records.", initial_records)

        # Handle empty DataFrame edge case
        if initial_records == 0:
            audit = TelematicsAuditMetrics(
                initial_records=0,
                valid_records=0,
                dropped_invalid_coordinates=0,
                dropped_invalid_speed=0,
                dropped_invalid_fuel=0,
                dropped_duplicate_timestamps=0,
                data_retention_rate_pct=0.0,
            ).to_dict()
            empty_df = pd.DataFrame(
                columns=[
                    self.vehicle_id_col,
                    self.timestamp_col,
                    self.latitude_col,
                    self.longitude_col,
                    self.speed_col,
                    self.fuel_col,
                    self.delta_time_col,
                    self.delta_speed_col,
                    self.delta_fuel_col,
                    "delta_t",
                    "delta_v",
                    "delta_l",
                ]
            )
            return empty_df, audit

        # Schema resolution & validation
        cols = self._resolve_column_names(df)
        self._validate_input_schema(df, cols)

        vid_col = cols["vehicle_id"]
        ts_col = cols["timestamp"]
        lat_col = cols["latitude"]
        lon_col = cols["longitude"]
        speed_col = cols["speed"]
        fuel_col = cols["fuel_level"]

        # ----------------------------------------------------------------------
        # Stage 1: Coordinate Validity
        # [-90, 90] for latitude, [-180, 180] for longitude, no NaNs
        # ----------------------------------------------------------------------
        lat_series = pd.to_numeric(df[lat_col], errors="coerce")
        lon_series = pd.to_numeric(df[lon_col], errors="coerce")

        coord_mask = (
            lat_series.notna()
            & lon_series.notna()
            & (lat_series >= self.min_lat)
            & (lat_series <= self.max_lat)
            & (lon_series >= self.min_lon)
            & (lon_series <= self.max_lon)
        )
        dropped_invalid_coordinates = int((~coord_mask).sum())
        df = df.loc[coord_mask]

        # ----------------------------------------------------------------------
        # Stage 2: Speed Sanitization
        # Speed in [0.0, 160.0] km/h
        # ----------------------------------------------------------------------
        speed_series = pd.to_numeric(df[speed_col], errors="coerce")
        speed_valid_mask = (
            speed_series.notna()
            & (speed_series >= self.min_speed)
            & (speed_series <= self.max_speed)
        )

        if self.drop_invalid_speed:
            dropped_invalid_speed = int((~speed_valid_mask).sum())
            df = df.loc[speed_valid_mask]
        else:
            # If flagging instead of dropping
            df["is_speed_valid"] = speed_valid_mask
            dropped_invalid_speed = 0

        # ----------------------------------------------------------------------
        # Stage 3: Fuel Reading Integrity
        # Fuel >= 0 and not in sentinel values (e.g. -999, -9999, -1)
        # ----------------------------------------------------------------------
        fuel_series = pd.to_numeric(df[fuel_col], errors="coerce")
        sentinel_arr = np.array(list(self.sentinel_fuel_values), dtype=np.float64)

        fuel_valid_mask = (
            fuel_series.notna()
            & (fuel_series >= 0.0)
            & (~fuel_series.isin(sentinel_arr))
        )
        dropped_invalid_fuel = int((~fuel_valid_mask).sum())
        df = df.loc[fuel_valid_mask]

        # ----------------------------------------------------------------------
        # Stage 4: Timestamp Parsing & Deduplication
        # Standard UTC datetime64[ns], sort by (vehicle_id, timestamp),
        # remove duplicate pings for the same vehicle.
        # ----------------------------------------------------------------------
        # Parse timestamp to UTC, then convert to standard datetime64[ns]
        utc_ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        valid_ts_mask = utc_ts.notna()
        if not valid_ts_mask.all():
            df = df.loc[valid_ts_mask]
            utc_ts = utc_ts.loc[valid_ts_mask]

        # Store as standard datetime64[ns] in target timezone or UTC
        if self.target_timezone and self.target_timezone.lower() in ("asia/kuala_lumpur", "myt", "utc+8"):
            myt_ts = utc_ts.dt.tz_convert("Asia/Kuala_Lumpur")
            df[ts_col] = myt_ts.dt.tz_localize(None).astype("datetime64[ns]")
        elif self.target_timezone:
            converted_ts = utc_ts.dt.tz_convert(self.target_timezone)
            df[ts_col] = converted_ts.dt.tz_localize(None).astype("datetime64[ns]")
        else:
            df[ts_col] = utc_ts.dt.tz_localize(None).astype("datetime64[ns]")

        # Stable sort by (vehicle_id, timestamp)
        df = df.sort_values(by=[vid_col, ts_col], ascending=[True, True], kind="mergesort")

        # Deduplicate pings for the same vehicle at the same timestamp
        duplicate_mask = df.duplicated(subset=[vid_col, ts_col], keep="first")
        dropped_duplicate_timestamps = int(duplicate_mask.sum())
        df = df.loc[~duplicate_mask].reset_index(drop=True)

        # ----------------------------------------------------------------------
        # Stage 5: Feature Engineering Precomputations (Vectorized)
        # ΔT = T_n - T_{n-1} (seconds)
        # ΔV = V_n - V_{n-1} (km/h)
        # ΔL = L_n - L_{n-1} (Liters)
        # Grouped by vehicle_id. Drop records where ΔT <= 0.
        # ----------------------------------------------------------------------
        n_rows = len(df)
        if n_rows > 0:
            v_ids = df[vid_col].to_numpy()
            same_vehicle = v_ids[:-1] == v_ids[1:]

            # ΔT (in seconds)
            dt_arr = np.full(n_rows, np.nan, dtype=np.float64)
            ts_vals = df[ts_col].to_numpy(dtype="datetime64[ns]")
            dt_arr[1:] = np.where(
                same_vehicle,
                (ts_vals[1:] - ts_vals[:-1]) / np.timedelta64(1, "s"),
                np.nan,
            )

            # ΔV (in km/h)
            dv_arr = np.full(n_rows, np.nan, dtype=np.float64)
            speed_vals = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64)
            dv_arr[1:] = np.where(
                same_vehicle,
                speed_vals[1:] - speed_vals[:-1],
                np.nan,
            )

            # ΔL (in Liters)
            dl_arr = np.full(n_rows, np.nan, dtype=np.float64)
            fuel_vals = pd.to_numeric(df[fuel_col], errors="coerce").to_numpy(dtype=np.float64)
            dl_arr[1:] = np.where(
                same_vehicle,
                fuel_vals[1:] - fuel_vals[:-1],
                np.nan,
            )

            # Assign precomputed features and aliases
            df[self.delta_time_col] = dt_arr
            df[self.delta_speed_col] = dv_arr
            df[self.delta_fuel_col] = dl_arr

            # Provide standard short aliases (delta_t, delta_v, delta_l)
            df["delta_t"] = dt_arr
            df["delta_v"] = dv_arr
            df["delta_l"] = dl_arr

            # Drop records where ΔT <= 0 (Note: first ping per vehicle has ΔT = NaN, preserved)
            non_positive_dt = df[self.delta_time_col] <= 0
            if non_positive_dt.any():
                df = df.loc[~non_positive_dt].reset_index(drop=True)
        else:
            # Empty after filters
            df[self.delta_time_col] = np.array([], dtype=np.float64)
            df[self.delta_speed_col] = np.array([], dtype=np.float64)
            df[self.delta_fuel_col] = np.array([], dtype=np.float64)
            df["delta_t"] = np.array([], dtype=np.float64)
            df["delta_v"] = np.array([], dtype=np.float64)
            df["delta_l"] = np.array([], dtype=np.float64)

        # Standardize canonical column names if aliases were resolved
        if lat_col != self.latitude_col:
            df[self.latitude_col] = df[lat_col]
        if lon_col != self.longitude_col:
            df[self.longitude_col] = df[lon_col]
        if speed_col != self.speed_col:
            df[self.speed_col] = df[speed_col]
        if fuel_col != self.fuel_col:
            df[self.fuel_col] = df[fuel_col]
        if vid_col != self.vehicle_id_col:
            df[self.vehicle_id_col] = df[vid_col]
        if ts_col != self.timestamp_col:
            df[self.timestamp_col] = df[ts_col]

        # ----------------------------------------------------------------------
        # Stage 6: Data Quality Audit Report
        # ----------------------------------------------------------------------
        valid_records = len(df)
        data_retention_rate_pct = (
            round((valid_records / initial_records) * 100.0, 2)
            if initial_records > 0
            else 0.0
        )

        audit_metrics = TelematicsAuditMetrics(
            initial_records=initial_records,
            valid_records=valid_records,
            dropped_invalid_coordinates=dropped_invalid_coordinates,
            dropped_invalid_speed=dropped_invalid_speed,
            dropped_invalid_fuel=dropped_invalid_fuel,
            dropped_duplicate_timestamps=dropped_duplicate_timestamps,
            data_retention_rate_pct=data_retention_rate_pct,
        )

        audit_dict = audit_metrics.to_dict()
        logger.info(
            "TelematicsCleaner completed: %d initial -> %d valid records (Retention: %.2f%%).",
            initial_records,
            valid_records,
            data_retention_rate_pct,
        )

        return df, audit_dict
