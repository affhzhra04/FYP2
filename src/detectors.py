"""Advanced Telematics Anomaly Detection Engine for Commercial IoT Fleets.

This module provides production-grade, vectorized anomaly detection including:
  1. EnhancedFuelDetector: Multi-condition state machine for fuel theft vs. refuel vs. sloshing.
  2. EnhancedMoonlightDetector: Spatiotemporal route boundary & driver night-shift roster checks.
  3. KinematicAggressionDetector: Longitudinal and lateral dynamics for harsh driving & idling.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from database.db_manager import TelematicsDBManager

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


# Earth mean radius in kilometers for Haversine calculations
EARTH_RADIUS_KM: float = 6371.0088


@dataclass(frozen=True)
class AnomalyEvent:
    """Canonical anomaly event representation for audit and database logging."""

    vehicle_id: str
    timestamp: str
    incident_type: str
    magnitude: float
    status: str = "OPEN"
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly event to serializable dictionary."""
        res = asdict(self)
        if res["details"] is None:
            res["details"] = {}
        return res


# ==============================================================================
# 1. Enhanced Fuel Detector (State Machine)
# ==============================================================================


class EnhancedFuelDetector:
    """Multi-condition state machine for IoT fuel theft and refuel event detection.

    Validates telemetry against:
      - Stationary vehicle state: speed == 0 and ignition == OFF.
      - Significant drop magnitude: drop exceeds configurable threshold (e.g., > 0.3L).
      - Multi-reading persistence: drop persists for >= 3 consecutive readings
        to reject ultrasonic acoustic sloshing noise.
      - Refuel classification: positive fuel surge > 5.0L triggers REFUEL event
        and resets baseline.

    Attributes:
        theft_threshold_liters: Minimum fuel drop (in Liters) to trigger theft alert.
        refuel_threshold_liters: Minimum positive jump (in Liters) to classify refuel.
        min_consecutive_readings: Number of consecutive readings a drop must persist.
        speed_tolerance_kmh: Maximum speed considered stationary (default 0.0).
    """

    def __init__(
        self,
        theft_threshold_liters: float = 0.3,
        refuel_threshold_liters: float = 5.0,
        min_consecutive_readings: int = 3,
        speed_tolerance_kmh: float = 0.0,
        fuel_col: str = "fuel_level",
        speed_col: str = "speed",
        ignition_col: str = "ignition",
        timestamp_col: str = "timestamp",
        vehicle_id_col: str = "vehicle_id",
    ) -> None:
        """Initialize the EnhancedFuelDetector with detection thresholds."""
        self.theft_threshold_liters = float(theft_threshold_liters)
        self.refuel_threshold_liters = float(refuel_threshold_liters)
        self.min_consecutive_readings = int(min_consecutive_readings)
        self.speed_tolerance_kmh = float(speed_tolerance_kmh)

        self.fuel_col = fuel_col
        self.speed_col = speed_col
        self.ignition_col = ignition_col
        self.timestamp_col = timestamp_col
        self.vehicle_id_col = vehicle_id_col

    def _is_ignition_off(self, ignition_val: Any) -> bool:
        """Evaluate whether an ignition reading indicates engine is OFF."""
        if pd.isna(ignition_val):
            # If ignition is not recorded, assume engine off only if stationary
            return True
        if isinstance(ignition_val, (bool, np.bool_)):
            return not bool(ignition_val)
        if isinstance(ignition_val, (int, float, np.integer, np.floating)):
            return float(ignition_val) == 0.0
        val_str = str(ignition_val).strip().upper()
        return val_str in ("0", "OFF", "FALSE", "NO", "N")

    def detect(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """Run fuel theft state machine over telematics time series.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned telematics DataFrame sorted by (vehicle_id, timestamp).

        Returns
        -------
        Tuple[pd.DataFrame, List[AnomalyEvent]]
            - Enriched DataFrame with 'fuel_event' and 'is_fuel_theft' columns.
            - List of AnomalyEvent instances ready for audit and database persistence.
        """
        if df.empty:
            df_out = df.copy()
            df_out["fuel_event"] = pd.Series(dtype="object")
            df_out["is_fuel_theft"] = pd.Series(dtype="bool")
            return df_out, []

        df_out = df.copy()
        df_out["fuel_event"] = "NORMAL"
        df_out["is_fuel_theft"] = False

        anomalies: List[AnomalyEvent] = []

        # Process trajectory per vehicle
        for vid, group in df_out.groupby(self.vehicle_id_col, sort=False):
            indices = group.index.to_numpy()
            n_points = len(indices)
            if n_points == 0:
                continue

            fuels = group[self.fuel_col].to_numpy(dtype=np.float64)
            speeds = group[self.speed_col].to_numpy(dtype=np.float64)
            timestamps = group[self.timestamp_col].to_numpy()

            has_ignition = self.ignition_col in group.columns
            ignitions = (
                group[self.ignition_col].to_numpy()
                if has_ignition
                else np.array([None] * n_points)
            )

            # State tracking variables
            baseline_fuel = fuels[0]
            consecutive_drop_count = 0
            candidate_drop_magnitude = 0.0
            theft_start_idx = indices[0]

            for i in range(n_points):
                curr_idx = indices[i]
                curr_fuel = fuels[i]
                curr_speed = speeds[i]
                curr_ts_str = str(timestamps[i])

                if np.isnan(curr_fuel):
                    continue

                # Check for REFUEL event first
                fuel_diff_from_baseline = curr_fuel - baseline_fuel
                step_fuel_diff = curr_fuel - fuels[i - 1] if i > 0 else 0.0

                if step_fuel_diff >= self.refuel_threshold_liters:
                    # Classify as REFUEL and reset baseline
                    df_out.at[curr_idx, "fuel_event"] = "REFUEL"
                    baseline_fuel = curr_fuel
                    consecutive_drop_count = 0
                    candidate_drop_magnitude = 0.0
                    anomalies.append(
                        AnomalyEvent(
                            vehicle_id=str(vid),
                            timestamp=curr_ts_str,
                            incident_type="REFUEL",
                            magnitude=round(float(step_fuel_diff), 2),
                            status="LOGGED",
                            details={"refuel_liters": round(float(step_fuel_diff), 2)},
                        )
                    )
                    continue

                # Check if vehicle is completely stationary (speed <= tol and ignition == OFF)
                is_stationary = (curr_speed <= self.speed_tolerance_kmh) and self._is_ignition_off(ignitions[i])

                if is_stationary:
                    drop_from_baseline = baseline_fuel - curr_fuel

                    if drop_from_baseline >= self.theft_threshold_liters:
                        consecutive_drop_count += 1
                        candidate_drop_magnitude = drop_from_baseline
                        if consecutive_drop_count == 1:
                            theft_start_idx = curr_idx

                        # Check if drop persists for at least min_consecutive_readings
                        if consecutive_drop_count >= self.min_consecutive_readings:
                            df_out.at[curr_idx, "fuel_event"] = "FUEL_THEFT"
                            df_out.at[curr_idx, "is_fuel_theft"] = True

                            # If just reached threshold, record formal anomaly
                            if consecutive_drop_count == self.min_consecutive_readings:
                                anomalies.append(
                                    AnomalyEvent(
                                        vehicle_id=str(vid),
                                        timestamp=str(df_out.at[theft_start_idx, self.timestamp_col]),
                                        incident_type="FUEL_THEFT",
                                        magnitude=round(float(-candidate_drop_magnitude), 2),
                                        status="OPEN",
                                        details={
                                            "drained_liters": round(float(candidate_drop_magnitude), 2),
                                            "consecutive_readings": consecutive_drop_count,
                                        },
                                    )
                                )
                    else:
                        # Slosh rebound: fuel returned to baseline, cancel false alert
                        if consecutive_drop_count < self.min_consecutive_readings:
                            consecutive_drop_count = 0
                            candidate_drop_magnitude = 0.0
                            # Adapt baseline slightly if sensor drifts upwards
                            if curr_fuel > baseline_fuel:
                                baseline_fuel = curr_fuel
                else:
                    # Vehicle is in motion or engine is running -> normal fuel consumption
                    baseline_fuel = curr_fuel
                    consecutive_drop_count = 0
                    candidate_drop_magnitude = 0.0

        logger.info(
            "EnhancedFuelDetector evaluated %d rows across %d vehicles -> %d anomalies detected.",
            len(df),
            df[self.vehicle_id_col].nunique() if not df.empty else 0,
            len(anomalies),
        )
        return df_out, anomalies


# ==============================================================================
# 2. Enhanced Moonlight Detector (Spatiotemporal & Roster Checks)
# ==============================================================================


class EnhancedMoonlightDetector:
    """Spatiotemporal route boundary & driver shift roster validation engine.

    Enforces fleet operational boundaries:
      1. Shift Schedule Verification: Queries the SQLite database manager
         (TelematicsDBManager) to check if a vehicle has an authorized night shift.
      2. Ghost Hours Enforcement: Late-night operating window (22:00–04:59).
         If a vehicle is active and NOT authorized for night shift, flags as
         UNAUTHORIZED_NIGHT_DRIVING.
      3. Geofence Boundary Check: Computes vectorized Haversine great-circle
         distance from central depot; flags trips exceeding corporate boundary.

    Attributes:
        db_manager: Instance of TelematicsDBManager.
        base_latitude: Latitude of central corporate depot in degrees.
        base_longitude: Longitude of central corporate depot in degrees.
        max_distance_km: Maximum authorized radius from depot in kilometers.
        ghost_hour_start: Start hour of ghost hours (default 22).
        ghost_hour_end: End hour of ghost hours (default 5, exclusive: 22:00–04:59).
    """

    DEFAULT_BASE_LAT: float = 3.1390
    DEFAULT_BASE_LON: float = 101.6869
    DEFAULT_MAX_DISTANCE_KM: float = 50.0

    def __init__(
        self,
        db_manager: Optional[TelematicsDBManager] = None,
        base_latitude: float = DEFAULT_BASE_LAT,
        base_longitude: float = DEFAULT_BASE_LON,
        max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
        ghost_hour_start: int = 22,
        ghost_hour_end: int = 5,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        timestamp_col: str = "timestamp",
        vehicle_id_col: str = "vehicle_id",
        speed_col: str = "speed",
    ) -> None:
        """Initialize EnhancedMoonlightDetector with geofence and roster DB."""
        self.db_manager = db_manager or TelematicsDBManager()
        self.base_latitude = float(base_latitude)
        self.base_longitude = float(base_longitude)
        self.max_distance_km = float(max_distance_km)
        self.ghost_hour_start = int(ghost_hour_start)
        self.ghost_hour_end = int(ghost_hour_end)

        self.lat_col = lat_col
        self.lon_col = lon_col
        self.timestamp_col = timestamp_col
        self.vehicle_id_col = vehicle_id_col
        self.speed_col = speed_col

        # In-memory cache for night-shift authorization to optimize bulk processing
        self._night_shift_cache: Dict[str, bool] = {}

    def is_authorized_night_shift(self, vehicle_id: str) -> bool:
        """Query database manager to verify if vehicle has an active night shift."""
        if vehicle_id not in self._night_shift_cache:
            is_auth = self.db_manager.is_night_shift(str(vehicle_id))
            self._night_shift_cache[vehicle_id] = is_auth
        return self._night_shift_cache[vehicle_id]

    @staticmethod
    def haversine_distance_vectorized(
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: float,
        lon2: float,
    ) -> np.ndarray:
        """Calculate vectorized great-circle distance in kilometers using the Haversine formula."""
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
        )
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return EARTH_RADIUS_KM * c

    def detect(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """Perform spatiotemporal geofence and Ghost Hours verification over telematics data.

        Parameters
        ----------
        df : pd.DataFrame
            Telematics DataFrame containing coordinates, timestamps, and vehicle IDs.

        Returns
        -------
        Tuple[pd.DataFrame, List[AnomalyEvent]]
            - Enriched DataFrame with 'distance_from_base_km', 'is_ghost_hours',
              'is_night_shift_authorized', 'is_unauthorized_night_driving', and 'is_geofence_breach'.
            - List of AnomalyEvent instances.
        """
        if df.empty:
            df_out = df.copy()
            df_out["distance_from_base_km"] = pd.Series(dtype="float64")
            df_out["is_ghost_hours"] = pd.Series(dtype="bool")
            df_out["is_night_shift_authorized"] = pd.Series(dtype="bool")
            df_out["is_unauthorized_night_driving"] = pd.Series(dtype="bool")
            df_out["is_geofence_breach"] = pd.Series(dtype="bool")
            return df_out, []

        df_out = df.copy()
        anomalies: List[AnomalyEvent] = []

        # 1. Vectorized Haversine Distance from Central Depot
        lats = df_out[self.lat_col].to_numpy(dtype=np.float64)
        lons = df_out[self.lon_col].to_numpy(dtype=np.float64)
        distances_km = self.haversine_distance_vectorized(
            lats, lons, self.base_latitude, self.base_longitude
        )
        df_out["distance_from_base_km"] = np.round(distances_km, 3)

        # Geofence breach condition
        geofence_breach_mask = distances_km > self.max_distance_km
        df_out["is_geofence_breach"] = geofence_breach_mask

        # 2. Ghost Hours (22:00 to 04:59)
        ts_series = pd.to_datetime(df_out[self.timestamp_col])
        hours = ts_series.dt.hour.to_numpy()
        ghost_hours_mask = (hours >= self.ghost_hour_start) | (hours < self.ghost_hour_end)
        df_out["is_ghost_hours"] = ghost_hours_mask

        # 3. Night-shift authorization check per vehicle from SQLite database
        vehicle_ids = df_out[self.vehicle_id_col].astype(str).to_numpy()
        unique_vids = np.unique(vehicle_ids)
        auth_map = {vid: self.is_authorized_night_shift(vid) for vid in unique_vids}

        is_auth_array = np.array([auth_map[vid] for vid in vehicle_ids], dtype=bool)
        df_out["is_night_shift_authorized"] = is_auth_array

        # Active driving during Ghost Hours without night shift authorization
        speed_array = df_out[self.speed_col].to_numpy(dtype=np.float64) if self.speed_col in df_out.columns else np.ones(len(df_out))
        moving_mask = speed_array > 0.0

        unauthorized_night_mask = ghost_hours_mask & (~is_auth_array) & moving_mask
        df_out["is_unauthorized_night_driving"] = unauthorized_night_mask

        # Generate anomaly event logs
        for idx in np.where(unauthorized_night_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="UNAUTHORIZED_NIGHT_DRIVING",
                    magnitude=float(speed_array[idx]),
                    status="OPEN",
                    details={
                        "hour": int(hours[idx]),
                        "speed_kmh": float(speed_array[idx]),
                        "authorized": False,
                    },
                )
            )

        for idx in np.where(geofence_breach_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="GEOFENCE_BREACH",
                    magnitude=float(round(distances_km[idx], 2)),
                    status="OPEN",
                    details={
                        "distance_km": float(round(distances_km[idx], 2)),
                        "max_allowed_km": self.max_distance_km,
                    },
                )
            )

        logger.info(
            "EnhancedMoonlightDetector evaluated %d rows -> %d anomalies (Ghost: %d, Geofence: %d).",
            len(df),
            len(anomalies),
            int(unauthorized_night_mask.sum()),
            int(geofence_breach_mask.sum()),
        )
        return df_out, anomalies


# ==============================================================================
# 3. Kinematic Aggression Detector (Dynamics & Driver Behavior)
# ==============================================================================


class KinematicAggressionDetector:
    """Vectorized kinematic aggression and vehicle operating behavior analyzer.

    Computes:
      - Longitudinal Acceleration: a = ΔV / ΔT in m/s².
        Flags harsh braking (a <= -3.0 m/s²) and harsh acceleration (a >= 2.5 m/s²).
      - Lateral Angular Velocity: omega = Δheading / ΔT in deg/s.
        Corrects for circular 360° modular wrapping to flag harsh cornering.
      - Speeding: Flags speed > 90.0 km/h.
      - Excessive Idling: Flags stationary duration > 300 seconds.

    Attributes:
        harsh_braking_threshold_mps2: Negative acceleration threshold in m/s² (default -3.0).
        harsh_accel_threshold_mps2: Positive acceleration threshold in m/s² (default 2.5).
        speed_limit_kmh: Maximum allowed speed threshold in km/h (default 90.0).
        harsh_cornering_threshold_dps: Angular velocity threshold in deg/s (default 25.0).
        excessive_idling_threshold_sec: Maximum permissible stationary duration in seconds (default 300.0).
        min_cornering_speed_kmh: Minimum speed required for harsh cornering alert (default 15.0).
    """

    KMH_TO_MPS: float = 1.0 / 3.6

    def __init__(
        self,
        harsh_braking_threshold_mps2: float = -3.0,
        harsh_accel_threshold_mps2: float = 2.5,
        speed_limit_kmh: float = 90.0,
        harsh_cornering_threshold_dps: float = 25.0,
        excessive_idling_threshold_sec: float = 300.0,
        min_cornering_speed_kmh: float = 15.0,
        speed_col: str = "speed",
        timestamp_col: str = "timestamp",
        vehicle_id_col: str = "vehicle_id",
        heading_col: str = "heading",
        delta_time_col: str = "delta_time",
        delta_speed_col: str = "delta_speed",
        ignition_col: str = "ignition",
    ) -> None:
        """Initialize KinematicAggressionDetector with behavioral thresholds."""
        self.harsh_braking_threshold_mps2 = float(harsh_braking_threshold_mps2)
        self.harsh_accel_threshold_mps2 = float(harsh_accel_threshold_mps2)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.harsh_cornering_threshold_dps = float(harsh_cornering_threshold_dps)
        self.excessive_idling_threshold_sec = float(excessive_idling_threshold_sec)
        self.min_cornering_speed_kmh = float(min_cornering_speed_kmh)

        self.speed_col = speed_col
        self.timestamp_col = timestamp_col
        self.vehicle_id_col = vehicle_id_col
        self.heading_col = heading_col
        self.delta_time_col = delta_time_col
        self.delta_speed_col = delta_speed_col
        self.ignition_col = ignition_col

    @staticmethod
    def compute_modular_heading_delta(heading_curr: np.ndarray, heading_prev: np.ndarray) -> np.ndarray:
        """Compute the shortest angular difference in degrees across the 360° boundary.

        Result is strictly bounded within [-180.0, 180.0].
        """
        diff = heading_curr - heading_prev
        return (diff + 180.0) % 360.0 - 180.0

    def detect(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """Vectorized evaluation of kinematic aggression and operating anomalies.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned telematics DataFrame sorted by (vehicle_id, timestamp).

        Returns
        -------
        Tuple[pd.DataFrame, List[AnomalyEvent]]
            - Enriched DataFrame with 'longitudinal_accel_mps2', 'angular_velocity_dps',
              'is_harsh_braking', 'is_harsh_accel', 'is_speeding', 'is_harsh_cornering',
              and 'is_excessive_idling'.
            - List of AnomalyEvent instances.
        """
        if df.empty:
            df_out = df.copy()
            df_out["longitudinal_accel_mps2"] = pd.Series(dtype="float64")
            df_out["angular_velocity_dps"] = pd.Series(dtype="float64")
            df_out["is_harsh_braking"] = pd.Series(dtype="bool")
            df_out["is_harsh_accel"] = pd.Series(dtype="bool")
            df_out["is_speeding"] = pd.Series(dtype="bool")
            df_out["is_harsh_cornering"] = pd.Series(dtype="bool")
            df_out["is_excessive_idling"] = pd.Series(dtype="bool")
            return df_out, []

        df_out = df.copy()
        n_rows = len(df_out)
        anomalies: List[AnomalyEvent] = []

        # Ensure vehicle trajectory grouping and boundaries
        v_ids = df_out[self.vehicle_id_col].to_numpy()
        same_vehicle = v_ids[:-1] == v_ids[1:] if n_rows > 1 else np.array([], dtype=bool)

        # ----------------------------------------------------------------------
        # 1. Delta Time (ΔT in seconds) and Delta Speed (ΔV in km/h)
        # ----------------------------------------------------------------------
        if self.delta_time_col in df_out.columns:
            dt_vals = df_out[self.delta_time_col].to_numpy(dtype=np.float64)
        else:
            ts_series = pd.to_datetime(df_out[self.timestamp_col])
            ts_ns = ts_series.to_numpy(dtype="datetime64[ns]")
            dt_vals = np.full(n_rows, np.nan, dtype=np.float64)
            if n_rows > 1:
                dt_vals[1:] = np.where(
                    same_vehicle,
                    (ts_ns[1:] - ts_ns[:-1]) / np.timedelta64(1, "s"),
                    np.nan,
                )

        speed_vals = df_out[self.speed_col].to_numpy(dtype=np.float64)
        if self.delta_speed_col in df_out.columns:
            dv_vals = df_out[self.delta_speed_col].to_numpy(dtype=np.float64)
        else:
            dv_vals = np.full(n_rows, np.nan, dtype=np.float64)
            if n_rows > 1:
                dv_vals[1:] = np.where(same_vehicle, speed_vals[1:] - speed_vals[:-1], np.nan)

        # ----------------------------------------------------------------------
        # 2. Longitudinal Acceleration (a = ΔV / ΔT in m/s²)
        # ----------------------------------------------------------------------
        accel_mps2 = np.full(n_rows, np.nan, dtype=np.float64)
        valid_dt_mask = (dt_vals > 0.0) & (~np.isnan(dt_vals))
        accel_mps2[valid_dt_mask] = (
            (dv_vals[valid_dt_mask] * self.KMH_TO_MPS) / dt_vals[valid_dt_mask]
        )
        df_out["longitudinal_accel_mps2"] = np.round(accel_mps2, 3)

        # Harsh braking and harsh acceleration
        harsh_braking_mask = accel_mps2 <= self.harsh_braking_threshold_mps2
        harsh_accel_mask = accel_mps2 >= self.harsh_accel_threshold_mps2

        df_out["is_harsh_braking"] = harsh_braking_mask
        df_out["is_harsh_accel"] = harsh_accel_mask

        # ----------------------------------------------------------------------
        # 3. Speeding (> 90 km/h)
        # ----------------------------------------------------------------------
        speeding_mask = speed_vals > self.speed_limit_kmh
        df_out["is_speeding"] = speeding_mask

        # ----------------------------------------------------------------------
        # 4. Lateral Angular Velocity (omega = Δheading / ΔT in deg/s) & Harsh Cornering
        # ----------------------------------------------------------------------
        angular_velocity = np.full(n_rows, np.nan, dtype=np.float64)
        has_heading = self.heading_col in df_out.columns

        if has_heading and n_rows > 1:
            headings = df_out[self.heading_col].to_numpy(dtype=np.float64)
            delta_headings = np.full(n_rows, np.nan, dtype=np.float64)
            delta_headings[1:] = np.where(
                same_vehicle,
                self.compute_modular_heading_delta(headings[1:], headings[:-1]),
                np.nan,
            )
            valid_heading_dt = valid_dt_mask & (~np.isnan(delta_headings))
            angular_velocity[valid_heading_dt] = np.abs(
                delta_headings[valid_heading_dt] / dt_vals[valid_heading_dt]
            )

        df_out["angular_velocity_dps"] = np.round(angular_velocity, 2)
        harsh_cornering_mask = (
            (angular_velocity >= self.harsh_cornering_threshold_dps)
            & (speed_vals >= self.min_cornering_speed_kmh)
        )
        df_out["is_harsh_cornering"] = harsh_cornering_mask

        # ----------------------------------------------------------------------
        # 5. Excessive Idling (> 300s stationary)
        # ----------------------------------------------------------------------
        excessive_idling_mask = np.zeros(n_rows, dtype=bool)

        for vid, group in df_out.groupby(self.vehicle_id_col, sort=False):
            grp_indices = group.index.to_numpy()
            grp_speeds = group[self.speed_col].to_numpy(dtype=np.float64)
            grp_dt = dt_vals[grp_indices]

            cumulative_idle_sec = 0.0
            for k in range(len(grp_indices)):
                idx = grp_indices[k]
                sp = grp_speeds[k]
                dt = grp_dt[k] if not np.isnan(grp_dt[k]) else 0.0

                if sp <= 0.0:
                    cumulative_idle_sec += dt
                    if cumulative_idle_sec > self.excessive_idling_threshold_sec:
                        excessive_idling_mask[idx] = True
                else:
                    cumulative_idle_sec = 0.0

        df_out["is_excessive_idling"] = excessive_idling_mask

        # ----------------------------------------------------------------------
        # 6. Event Serialization
        # ----------------------------------------------------------------------
        for idx in np.where(harsh_braking_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="HARSH_BRAKING",
                    magnitude=float(round(accel_mps2[idx], 2)),
                    status="OPEN",
                    details={"accel_mps2": float(round(accel_mps2[idx], 2))},
                )
            )

        for idx in np.where(harsh_accel_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="HARSH_ACCELERATION",
                    magnitude=float(round(accel_mps2[idx], 2)),
                    status="OPEN",
                    details={"accel_mps2": float(round(accel_mps2[idx], 2))},
                )
            )

        for idx in np.where(speeding_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="OVERSPEEDING",
                    magnitude=float(round(speed_vals[idx], 2)),
                    status="OPEN",
                    details={"speed_kmh": float(round(speed_vals[idx], 2))},
                )
            )

        for idx in np.where(harsh_cornering_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="HARSH_CORNERING",
                    magnitude=float(round(angular_velocity[idx], 2)),
                    status="OPEN",
                    details={"angular_velocity_dps": float(round(angular_velocity[idx], 2))},
                )
            )

        for idx in np.where(excessive_idling_mask)[0]:
            anomalies.append(
                AnomalyEvent(
                    vehicle_id=str(df_out.iloc[idx][self.vehicle_id_col]),
                    timestamp=str(df_out.iloc[idx][self.timestamp_col]),
                    incident_type="EXCESSIVE_IDLING",
                    magnitude=float(round(self.excessive_idling_threshold_sec, 2)),
                    status="OPEN",
                    details={"threshold_sec": self.excessive_idling_threshold_sec},
                )
            )

        logger.info(
            "KinematicAggressionDetector evaluated %d rows -> %d events (Braking: %d, Accel: %d, Speeding: %d, Cornering: %d, Idling: %d).",
            len(df),
            len(anomalies),
            int(harsh_braking_mask.sum()),
            int(harsh_accel_mask.sum()),
            int(speeding_mask.sum()),
            int(harsh_cornering_mask.sum()),
            int(excessive_idling_mask.sum()),
        )
        return df_out, anomalies
