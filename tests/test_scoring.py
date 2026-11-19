"""Comprehensive Test Suite for Driver Safety Scoring and Risk Classification Engine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.detectors import AnomalyEvent
from src.scoring import RiskScoreReport, RiskScoringEngine, RiskTier


@pytest.fixture
def scoring_engine() -> RiskScoringEngine:
    """Fixture providing a default RiskScoringEngine instance."""
    return RiskScoringEngine()


# ==============================================================================
# 1. Configurable Weight Management Tests
# ==============================================================================


def test_default_weights_and_retrieval(scoring_engine: RiskScoringEngine) -> None:
    """Verify default weight configuration and retrieval."""
    weights = scoring_engine.get_weights()
    assert weights["fuel_theft"] == 5.0
    assert weights["moonlighting"] == 4.0
    assert weights["speeding"] == 3.0
    assert weights["harsh_driving"] == 2.0
    assert weights["idling"] == 1.0


def test_dynamic_weight_update_valid(scoring_engine: RiskScoringEngine) -> None:
    """Verify updating category weights within valid range [1.0, 5.0]."""
    scoring_engine.set_weight("speeding", 4.5)
    assert scoring_engine.get_weights()["speeding"] == 4.5

    scoring_engine.set_weights({"fuel_theft": 4.0, "idling": 2.5})
    assert scoring_engine.get_weights()["fuel_theft"] == 4.0
    assert scoring_engine.get_weights()["idling"] == 2.5


def test_weight_boundaries_validation(scoring_engine: RiskScoringEngine) -> None:
    """Verify that setting weight outside [1.0, 5.0] raises ValueError."""
    with pytest.raises(ValueError, match="between 1.0 and 5.0"):
        scoring_engine.set_weight("speeding", 0.5)  # Under 1.0

    with pytest.raises(ValueError, match="between 1.0 and 5.0"):
        scoring_engine.set_weight("speeding", 5.5)  # Over 5.0

    with pytest.raises(KeyError, match="Unknown infraction category"):
        scoring_engine.set_weight("invalid_category", 3.0)


# ==============================================================================
# 2. Risk Classification Tiers Tests
# ==============================================================================


@pytest.mark.parametrize(
    "score,expected_tier",
    [
        (100.0, RiskTier.EXCELLENT),
        (90.0, RiskTier.EXCELLENT),
        (89.9, RiskTier.LOW_RISK),
        (75.0, RiskTier.LOW_RISK),
        (74.9, RiskTier.MODERATE_RISK),
        (50.0, RiskTier.MODERATE_RISK),
        (49.9, RiskTier.HIGH_RISK),
        (25.0, RiskTier.HIGH_RISK),
        (24.9, RiskTier.CRITICAL),
        (0.0, RiskTier.CRITICAL),
    ],
)
def test_risk_tier_classification_boundaries(
    scoring_engine: RiskScoringEngine,
    score: float,
    expected_tier: RiskTier,
) -> None:
    """Verify boundary thresholds for 5-tier safety classification."""
    assert scoring_engine.classify_tier(score) == expected_tier


# ==============================================================================
# 3. Continuous Score Calculation & Bounding Tests
# ==============================================================================


def test_baseline_score_with_zero_infractions(scoring_engine: RiskScoringEngine) -> None:
    """Verify baseline 100.00 score when no infractions occur."""
    report = scoring_engine.calculate_score({}, vehicle_id="V_CLEAN")

    assert report.safety_score == 100.00
    assert report.risk_tier == RiskTier.EXCELLENT.value
    assert report.total_deductions == 0.0
    assert report.compounding_applied is False
    assert report.vehicle_id == "V_CLEAN"


def test_score_bounding_at_zero(scoring_engine: RiskScoringEngine) -> None:
    """Verify that extreme deductions are strictly bounded at 0.00 (never negative)."""
    # Massive infractions totaling > 100 points
    catastrophic_counts = {
        "fuel_theft": 10,
        "moonlighting": 10,
        "speeding": 10,
        "harsh_driving": 10,
        "idling": 10,
    }
    report = scoring_engine.calculate_score(catastrophic_counts, vehicle_id="V_CRITICAL")

    assert report.safety_score == 0.00
    assert report.risk_tier == RiskTier.CRITICAL.value
    assert report.total_deductions > 100.0


# ==============================================================================
# 4. Non-Linear Compounding Penalties Tests
# ==============================================================================


def test_isolated_infractions_no_compounding(scoring_engine: RiskScoringEngine) -> None:
    """Verify linear penalties when no high-risk co-occurrence occurs."""
    # Isolated speeding only
    report = scoring_engine.calculate_score({"speeding": 2}, vehicle_id="V_SPEED")

    assert report.compounding_applied is False
    assert report.compounding_multiplier == 1.0
    # Base speeding penalty: 2 * 2.0 * (3.0/3.0) = 4.0 points
    assert report.breakdown["speeding"] == 4.0
    assert report.total_deductions == 4.0
    assert report.safety_score == 96.00
    assert report.risk_tier == RiskTier.EXCELLENT.value


def test_compounding_multiplier_on_co_occurrence(scoring_engine: RiskScoringEngine) -> None:
    """Verify that speeding during a night-time moonlighting trip triggers the 1.5x interaction multiplier."""
    # Co-occurring speeding + moonlighting
    co_occurring_counts = {
        "speeding": 2,
        "moonlighting": 1,
    }
    report = scoring_engine.calculate_score(co_occurring_counts, vehicle_id="V_COMPOUND")

    assert report.compounding_applied is True
    assert report.compounding_multiplier == 1.5
    assert len(report.compounded_interactions) >= 1

    # Linear moonlighting: 1 * 3.5 * (4/3) = 4.67 pts. With 1.5x multiplier: ~7.00 pts
    # Linear speeding: 2 * 2.0 * (3/3) = 4.00 pts. With 1.5x multiplier: 6.00 pts
    assert report.breakdown["speeding"] == 6.0
    assert pytest.approx(report.breakdown["moonlighting"], abs=0.1) == 7.00
    assert report.total_deductions > 10.0


def test_dataframe_row_level_co_occurrence(scoring_engine: RiskScoringEngine) -> None:
    """Verify detection of row-level co-occurring violations (e.g. speeding during unauthorized night driving)."""
    df_violations = pd.DataFrame(
        {
            "vehicle_id": ["8_9_8", "8_9_8"],
            "is_speeding": [True, False],
            "is_unauthorized_night_driving": [True, False],  # Row 0: Speeding while Moonlighting!
            "is_fuel_theft": [False, False],
            "is_harsh_braking": [False, False],
            "is_harsh_accel": [False, False],
            "is_harsh_cornering": [False, False],
            "is_excessive_idling": [False, False],
        }
    )

    report = scoring_engine.calculate_score(df_violations)

    assert report.compounding_applied is True
    assert "SPEEDING_DURING_MOONLIGHTING" in report.compounded_interactions
    assert report.breakdown["speeding"] == 3.0  # 1 * 2.0 * (3/3) * 1.5 = 3.0
    assert pytest.approx(report.breakdown["moonlighting"], abs=0.1) == 7.0  # 1 * 3.5 * (4/3) * 1.5 = 7.0


# ==============================================================================
# 5. Explainable Breakdown & UI Reporting Tests
# ==============================================================================


def test_explainable_breakdown_structures(scoring_engine: RiskScoringEngine) -> None:
    """Verify structured dictionary and DataFrame outputs for UI dashboard display."""
    counts = {
        "fuel_theft": 1,
        "harsh_driving": 3,
        "idling": 2,
    }
    report = scoring_engine.calculate_score(counts, vehicle_id="TRUCK_421")

    # Dictionary representation
    rep_dict = report.to_dict()
    assert isinstance(rep_dict, dict)
    assert "safety_score" in rep_dict
    assert "risk_tier" in rep_dict
    assert "breakdown" in rep_dict
    assert rep_dict["vehicle_id"] == "TRUCK_421"

    # Tabular DataFrame representation for UI
    df_breakdown = report.to_dataframe()
    assert isinstance(df_breakdown, pd.DataFrame)
    assert set(df_breakdown.columns) == {"infraction_category", "incident_count", "points_deducted"}
    assert len(df_breakdown) == 5  # 5 categories


# ==============================================================================
# 6. Integration: AnomalyEvent Sequence & Fleet Scoring
# ==============================================================================


def test_score_calculation_from_anomaly_events(scoring_engine: RiskScoringEngine) -> None:
    """Verify scoring directly from a list of AnomalyEvent instances produced by detectors."""
    events = [
        AnomalyEvent(
            vehicle_id="8_9_8",
            timestamp="2026-03-01 10:00:00",
            incident_type="OVERSPEEDING",
            magnitude=105.0,
        ),
        AnomalyEvent(
            vehicle_id="8_9_8",
            timestamp="2026-03-01 10:05:00",
            incident_type="HARSH_BRAKING",
            magnitude=-4.2,
        ),
    ]

    report = scoring_engine.calculate_score(events)

    assert report.vehicle_id == "8_9_8"
    assert report.infraction_counts["speeding"] == 1
    assert report.infraction_counts["harsh_driving"] == 1
    assert report.safety_score < 100.0


def test_fleet_scoring_aggregation(scoring_engine: RiskScoringEngine) -> None:
    """Verify fleet-wide batch scoring across multiple commercial vehicles."""
    df_fleet = pd.DataFrame(
        {
            "vehicle_id": ["V_SAFE", "V_RISKY", "V_RISKY"],
            "is_speeding": [False, True, True],
            "is_unauthorized_night_driving": [False, True, False],
            "is_fuel_theft": [False, False, False],
            "is_harsh_braking": [False, True, False],
            "is_harsh_accel": [False, False, False],
            "is_harsh_cornering": [False, False, False],
            "is_excessive_idling": [False, False, False],
        }
    )

    fleet_summary = scoring_engine.calculate_fleet_scores(df_fleet)

    assert isinstance(fleet_summary, pd.DataFrame)
    assert len(fleet_summary) == 2  # V_SAFE and V_RISKY
    assert "safety_score" in fleet_summary.columns
    assert "risk_tier" in fleet_summary.columns

    # V_SAFE should have a higher safety score than V_RISKY
    safe_row = fleet_summary[fleet_summary["vehicle_id"] == "V_SAFE"].iloc[0]
    risky_row = fleet_summary[fleet_summary["vehicle_id"] == "V_RISKY"].iloc[0]
    assert safe_row["safety_score"] > risky_row["safety_score"]
    assert safe_row["risk_tier"] == RiskTier.EXCELLENT.value
