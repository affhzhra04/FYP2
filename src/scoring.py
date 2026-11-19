"""Driver Safety Scoring and Non-Linear Risk Classification Engine for FYP2 Telematics.

This module provides production-grade risk assessment and score calculation:
  1. Configurable weight management (weights bounded between 1.0 and 5.0).
  2. Non-linear interaction multipliers (e.g. 1.5x) for co-occurring high-risk infractions.
  3. Continuous Driver Safety Score (DSS) computed from baseline 100.00, bounded [0.00, 100.00].
  4. 5-tier safety risk classification (Excellent, Low, Moderate, High, Critical).
  5. Explainable penalty breakdown structured for transparent UI dashboard display.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from typing import Tuple, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.detectors import AnomalyEvent

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


class RiskTier(str, Enum):
    """Canonical risk classification tiers for fleet safety auditing."""

    EXCELLENT = "Excellent (Compliant)"
    LOW_RISK = "Low Risk"
    MODERATE_RISK = "Moderate Risk"
    HIGH_RISK = "High Risk"
    CRITICAL = "Critical (Immediate Audit Required)"


@dataclass(frozen=True)
class RiskScoreReport:
    """Comprehensive explainable audit report for Driver Safety Score (DSS)."""

    safety_score: float
    risk_tier: str
    baseline_score: float = 100.0
    total_deductions: float = 0.0
    compounding_applied: bool = False
    compounding_multiplier: float = 1.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    infraction_counts: Dict[str, int] = field(default_factory=dict)
    compounded_interactions: List[str] = field(default_factory=list)
    recommendation: str = ""
    vehicle_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standard serializable dictionary."""
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert report breakdown into a tabular DataFrame for UI presentation."""
        rows = []
        for cat, deduction in self.breakdown.items():
            count = self.infraction_counts.get(cat, 0)
            rows.append(
                {
                    "infraction_category": cat,
                    "incident_count": count,
                    "points_deducted": round(deduction, 2),
                }
            )
        df_breakdown = pd.DataFrame(rows)
        return df_breakdown


class RiskScoringEngine:
    """Production-grade scoring engine with non-linear penalties and risk tier classification.

    Attributes:
        weights (Dict[str, float]): Dynamic weight scaling for each category (bounded [1.0, 5.0]).
        base_penalties (Dict[str, float]): Base point deduction per incident.
        compound_multiplier (float): Interaction penalty multiplier for co-occurring high-risk events.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "fuel_theft": 5.0,     # Highest severity: deliberate theft/pilferage
        "moonlighting": 4.0,   # Severe: unauthorized vehicle usage/geofence breach
        "speeding": 3.0,       # High safety risk: highway speeding
        "harsh_driving": 2.0,  # Moderate: aggressive braking, acceleration, cornering
        "idling": 1.0,         # Minor: excessive idling / fuel wastage
    }

    DEFAULT_BASE_PENALTIES: Dict[str, float] = {
        "fuel_theft": 5.0,     # 5.0 base pts * weight
        "moonlighting": 3.5,   # 3.5 base pts * weight
        "speeding": 2.0,       # 2.0 base pts * weight
        "harsh_driving": 1.5,  # 1.5 base pts * weight
        "idling": 1.0,         # 1.0 base pts * weight
    }

    # Incident type mappings to canonical categories
    INCIDENT_TYPE_MAP: Dict[str, str] = {
        "FUEL_THEFT": "fuel_theft",
        "UNAUTHORIZED_NIGHT_DRIVING": "moonlighting",
        "GEOFENCE_BREACH": "moonlighting",
        "MOONLIGHTING": "moonlighting",
        "OVERSPEEDING": "speeding",
        "SPEEDING": "speeding",
        "HARSH_BRAKING": "harsh_driving",
        "HARSH_ACCELERATION": "harsh_driving",
        "HARSH_CORNERING": "harsh_driving",
        "EXCESSIVE_IDLING": "idling",
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        base_penalties: Optional[Dict[str, float]] = None,
        compound_multiplier: float = 1.5,
    ) -> None:
        """Initialize the RiskScoringEngine.

        Parameters
        ----------
        weights : Optional[Dict[str, float]], optional
            Custom category weights (each in range [1.0, 5.0]).
        base_penalties : Optional[Dict[str, float]], optional
            Custom base penalty points per incident type.
        compound_multiplier : float, optional
            Interaction compounding multiplier (default 1.5x).

        Raises
        ------
        ValueError
            If any weight is outside [1.0, 5.0] or compound_multiplier < 1.0.
        """
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.set_weights(weights)

        self.base_penalties = dict(self.DEFAULT_BASE_PENALTIES)
        if base_penalties:
            for k, v in base_penalties.items():
                if v < 0:
                    raise ValueError(f"Base penalty for '{k}' must be non-negative: {v}")
                self.base_penalties[k] = float(v)

        if compound_multiplier < 1.0:
            raise ValueError(f"Compound multiplier must be >= 1.0, got: {compound_multiplier}")
        self.compound_multiplier = float(compound_multiplier)

    def set_weight(self, category: str, weight: float) -> None:
        """Set dynamic weight scaling for a specific category.

        Parameters
        ----------
        category : str
            Infraction category (fuel_theft, moonlighting, speeding, harsh_driving, idling).
        weight : float
            Numeric weight strictly bounded within [1.0, 5.0].

        Raises
        ------
        KeyError
            If category is invalid.
        ValueError
            If weight is not within [1.0, 5.0].
        """
        if category not in self.DEFAULT_WEIGHTS:
            raise KeyError(
                f"Unknown infraction category '{category}'. "
                f"Available categories: {list(self.DEFAULT_WEIGHTS.keys())}"
            )
        w_val = float(weight)
        if not (1.0 <= w_val <= 5.0):
            raise ValueError(
                f"Weight for '{category}' must be between 1.0 and 5.0 (inclusive). Got: {weight}"
            )
        self.weights[category] = w_val
        logger.info("Updated weight for category '%s' to %.2f", category, w_val)

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Bulk update dynamic weight scaling."""
        for cat, w in weights.items():
            self.set_weight(cat, w)

    def get_weights(self) -> Dict[str, float]:
        """Retrieve current configured weights dictionary."""
        return dict(self.weights)

    @staticmethod
    def classify_tier(score: float) -> RiskTier:
        """Automatically categorize Driver Safety Score into risk classification tier.

        Tiers:
          - 90.00 – 100.00: Excellent (Compliant)
          - 75.00 – 89.99: Low Risk
          - 50.00 – 74.99: Moderate Risk
          - 25.00 – 49.99: High Risk
          - 0.00 – 24.99: Critical (Immediate Audit Required)
        """
        s = round(float(score), 2)
        if s >= 90.0:
            return RiskTier.EXCELLENT
        if s >= 75.0:
            return RiskTier.LOW_RISK
        if s >= 50.0:
            return RiskTier.MODERATE_RISK
        if s >= 25.0:
            return RiskTier.HIGH_RISK
        return RiskTier.CRITICAL

    @staticmethod
    def get_tier_recommendation(tier: RiskTier) -> str:
        """Provide automated operational recommendations corresponding to risk tier."""
        recommendations = {
            RiskTier.EXCELLENT: "Driver operates within full compliance. Commended for safe driving.",
            RiskTier.LOW_RISK: "Minor isolated infractions. Recommend automated safety coaching reminder.",
            RiskTier.MODERATE_RISK: "Elevated risk profile detected. Schedule supervisory review and route audit.",
            RiskTier.HIGH_RISK: "Severe risk patterns identified. Issue formal warning and mandatory refresher training.",
            RiskTier.CRITICAL: "CRITICAL: Multiple high-risk violations. Immediate operational suspension and forensic audit required.",
        }
        return recommendations.get(tier, "Review driver telemetry logs.")

    def _extract_counts_from_events(
        self,
        events: Sequence[Union[AnomalyEvent, Dict[str, Any]]],
    ) -> Dict[str, int]:
        """Aggregate infraction counts categorized from a sequence of AnomalyEvent objects."""
        counts = {cat: 0 for cat in self.DEFAULT_WEIGHTS}
        for ev in events:
            if isinstance(ev, AnomalyEvent):
                i_type = ev.incident_type.upper()
            elif isinstance(ev, dict):
                i_type = str(ev.get("incident_type", "")).upper()
            else:
                continue

            category = self.INCIDENT_TYPE_MAP.get(i_type)
            if category and category in counts:
                counts[category] += 1
        return counts

    def _extract_counts_and_interactions_from_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, int], List[str]]:
        """Vectorized extraction of infraction counts and co-occurring interaction patterns from DataFrame."""
        counts = {cat: 0 for cat in self.DEFAULT_WEIGHTS}
        interactions: List[str] = []

        if df.empty:
            return counts, interactions

        # Identify boolean flags in dataframe
        has_fuel_theft = (
            df["is_fuel_theft"].to_numpy(dtype=bool)
            if "is_fuel_theft" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        has_night = (
            df["is_unauthorized_night_driving"].to_numpy(dtype=bool)
            if "is_unauthorized_night_driving" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        has_geofence = (
            df["is_geofence_breach"].to_numpy(dtype=bool)
            if "is_geofence_breach" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        has_moonlighting = has_night | has_geofence

        has_speeding = (
            df["is_speeding"].to_numpy(dtype=bool)
            if "is_speeding" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        has_braking = (
            df["is_harsh_braking"].to_numpy(dtype=bool)
            if "is_harsh_braking" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        has_accel = (
            df["is_harsh_accel"].to_numpy(dtype=bool)
            if "is_harsh_accel" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        has_corner = (
            df["is_harsh_cornering"].to_numpy(dtype=bool)
            if "is_harsh_cornering" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        has_harsh = has_braking | has_accel | has_corner

        has_idling = (
            df["is_excessive_idling"].to_numpy(dtype=bool)
            if "is_excessive_idling" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        counts["fuel_theft"] = int(has_fuel_theft.sum())
        counts["moonlighting"] = int(has_moonlighting.sum())
        counts["speeding"] = int(has_speeding.sum())
        counts["harsh_driving"] = int(has_harsh.sum())
        counts["idling"] = int(has_idling.sum())

        # Check row-level co-occurrence interactions
        # 1. Speeding while Moonlighting (e.g. speeding during unauthorized night trip)
        speeding_moonlighting = has_speeding & has_moonlighting
        if speeding_moonlighting.any():
            interactions.append("SPEEDING_DURING_MOONLIGHTING")

        # 2. Fuel Theft during Moonlighting
        theft_moonlighting = has_fuel_theft & has_moonlighting
        if theft_moonlighting.any():
            interactions.append("FUEL_THEFT_DURING_MOONLIGHTING")

        # 3. Harsh driving while Speeding (extreme reckless operation)
        speeding_harsh = has_speeding & has_harsh
        if speeding_harsh.any():
            interactions.append("RECKLESS_HARSH_SPEEDING")

        return counts, interactions

    def calculate_score(
        self,
        data: Union[pd.DataFrame, Sequence[Union[AnomalyEvent, Dict[str, Any]]], Dict[str, int]],
        vehicle_id: Optional[str] = None,
    ) -> RiskScoreReport:
        """Calculate continuous Driver Safety Score (DSS) and explainable risk breakdown.

        Parameters
        ----------
        data : Union[pd.DataFrame, Sequence[Union[AnomalyEvent, Dict[str, Any]]], Dict[str, int]]
            Input data source. Can be:
              - Enriched telematics DataFrame from detectors.
              - Sequence of AnomalyEvent instances or dictionaries.
              - Pre-aggregated dictionary of infraction counts.
        vehicle_id : Optional[str], optional
            Identifier of vehicle or driver for audit tracking.

        Returns
        -------
        RiskScoreReport
            Complete explainable safety score breakdown.
        """
        interactions: List[str] = []

        if isinstance(data, pd.DataFrame):
            counts, df_interactions = self._extract_counts_and_interactions_from_dataframe(data)
            interactions.extend(df_interactions)
            if vehicle_id is None and "vehicle_id" in data.columns and not data.empty:
                vehicle_id = str(data["vehicle_id"].iloc[0])
        elif isinstance(data, (list, tuple)):
            counts = self._extract_counts_from_events(data)
            if vehicle_id is None and data and hasattr(data[0], "vehicle_id"):
                vehicle_id = getattr(data[0], "vehicle_id")
        elif isinstance(data, dict):
            counts = {cat: int(data.get(cat, 0)) for cat in self.DEFAULT_WEIGHTS}
        else:
            raise TypeError(
                f"Unsupported data format for risk scoring: {type(data).__name__}. "
                "Expected pd.DataFrame, Sequence[AnomalyEvent], or Dict[str, int]."
            )

        # High-risk co-occurrence check at aggregate level if not detected at row level
        if "SPEEDING_DURING_MOONLIGHTING" not in interactions and (counts["speeding"] > 0 and counts["moonlighting"] > 0):
            interactions.append("SPEEDING_AND_MOONLIGHTING_CO_OCCURRENCE")

        if "FUEL_THEFT_DURING_MOONLIGHTING" not in interactions and (counts["fuel_theft"] > 0 and counts["moonlighting"] > 0):
            interactions.append("FUEL_THEFT_AND_MOONLIGHTING_CO_OCCURRENCE")

        compounding_active = len(interactions) > 0
        active_multiplier = self.compound_multiplier if compounding_active else 1.0

        # Calculate exact point deductions per category
        breakdown: Dict[str, float] = {}
        total_deduction = 0.0

        for cat, count in counts.items():
            base_p = self.base_penalties[cat]
            w = self.weights[cat]
            subtotal = count * base_p * (w / 3.0)  # normalized relative to baseline weight=3

            # Apply interaction multiplier if category is part of active co-occurrence
            if compounding_active:
                if cat in ("speeding", "moonlighting") and any("MOONLIGHTING" in s for s in interactions):
                    subtotal *= active_multiplier
                elif cat == "fuel_theft" and any("THEFT" in s for s in interactions):
                    subtotal *= active_multiplier
                elif cat == "harsh_driving" and any("RECKLESS" in s for s in interactions):
                    subtotal *= active_multiplier

            rounded_subtotal = round(subtotal, 2)
            breakdown[cat] = rounded_subtotal
            total_deduction += rounded_subtotal

        total_deduction = round(total_deduction, 2)

        # Compute continuous DSS bounded strictly between 0.00 and 100.00
        raw_score = 100.0 - total_deduction
        safety_score = round(float(np.clip(raw_score, 0.00, 100.00)), 2)

        risk_tier = self.classify_tier(safety_score)
        recommendation = self.get_tier_recommendation(risk_tier)

        report = RiskScoreReport(
            safety_score=safety_score,
            risk_tier=risk_tier.value,
            baseline_score=100.0,
            total_deductions=total_deduction,
            compounding_applied=compounding_active,
            compounding_multiplier=active_multiplier,
            breakdown=breakdown,
            infraction_counts=counts,
            compounded_interactions=interactions,
            recommendation=recommendation,
            vehicle_id=vehicle_id,
        )

        logger.info(
            "Computed DSS: %.2f | Tier: %s | Deductions: %.2f (Compounding: %s) for %s",
            safety_score,
            risk_tier.value,
            total_deduction,
            compounding_active,
            vehicle_id or "fleet",
        )
        return report

    def calculate_fleet_scores(
        self,
        df: pd.DataFrame,
        vehicle_id_col: str = "vehicle_id",
    ) -> pd.DataFrame:
        """Calculate safety scores for an entire commercial fleet grouped by vehicle.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched telematics DataFrame with anomaly flags across multiple vehicles.
        vehicle_id_col : str, optional
            Column name grouping vehicles. Defaults to 'vehicle_id'.

        Returns
        -------
        pd.DataFrame
            Tabular fleet summary DataFrame containing vehicle scores, tiers, and breakdowns.
        """
        if df.empty or vehicle_id_col not in df.columns:
            return pd.DataFrame()

        fleet_reports: List[Dict[str, Any]] = []

        for vid, group in df.groupby(vehicle_id_col, sort=False):
            report = self.calculate_score(group, vehicle_id=str(vid))
            rep_dict = {
                "vehicle_id": str(vid),
                "safety_score": report.safety_score,
                "risk_tier": report.risk_tier,
                "total_deductions": report.total_deductions,
                "compounding_applied": report.compounding_applied,
                "compounding_multiplier": report.compounding_multiplier,
            }
            # Add category deduction breakdown columns
            for cat, pts in report.breakdown.items():
                rep_dict[f"deduction_{cat}"] = pts
            for cat, cnt in report.infraction_counts.items():
                rep_dict[f"count_{cat}"] = cnt

            fleet_reports.append(rep_dict)

        df_fleet = pd.DataFrame(fleet_reports)
        df_fleet = df_fleet.sort_values(by="safety_score", ascending=True).reset_index(drop=True)
        return df_fleet
