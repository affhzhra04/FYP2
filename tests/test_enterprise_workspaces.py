"""Enterprise Test Suite for Timezone Enforcement, Workspaces, and Replay Features."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Generator
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from database.db_manager import TelematicsDBManager
from src.services.data_service import TelematicsDataService
from src.utils.timezone import (
    MYT_TIMEZONE_STR,
    MYT_TZ,
    is_ghost_hours_myt,
    now_myt,
    to_myt_datetime,
    to_myt_series,
    to_myt_str,
)


@pytest.fixture
def mock_telematics_service() -> Generator[TelematicsDataService, None, None]:
    """Provide a populated test TelematicsDataService."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_telematics.db"
        mgr = TelematicsDBManager(db_path=db_path, auto_init=True, auto_seed=True)

        # Seed additional vehicle with telemetry and anomalies
        mgr.add_vehicle("1_10_7", "BXY-1070", "Isuzu Giga", 400.0)
        mgr.add_driver_schedule("DRV_107", "1_10_7", "08:00:00", "17:00:00", False)

        pings = pd.DataFrame([
            {
                "vehicle_id": "1_10_7",
                "timestamp": "2024-01-10 10:00:00",
                "latitude": 3.1390,
                "longitude": 101.6869,
                "speed": 85.0,
                "fuel_level": 350.0,
                "heading": 90.0,
                "ignition": 1,
            },
            {
                "vehicle_id": "1_10_7",
                "timestamp": "2024-01-10 10:15:00",
                "latitude": 3.1420,
                "longitude": 101.6900,
                "speed": 0.0,
                "fuel_level": 345.0,
                "heading": 90.0,
                "ignition": 0,
            },
        ])
        mgr.save_telematics_pings(pings)

        mgr.log_anomaly("1_10_7", "2024-01-10 10:15:00", "FUEL_THEFT", -5.0, status="OPEN")
        mgr.log_anomaly("1_10_7", "2024-01-10 23:30:00", "UNAUTHORIZED_NIGHT_DRIVING", 45.0, status="OPEN")

        service = TelematicsDataService(db_path=db_path)
        yield service


def test_now_myt_timezone() -> None:
    """Verify now_myt produces timezone-aware datetime in Asia/Kuala_Lumpur."""
    dt = now_myt()
    assert dt.tzinfo is not None
    assert str(dt.tzinfo) == MYT_TIMEZONE_STR
    assert dt.utcoffset().total_seconds() == 28800  # +8 hours


def test_to_myt_datetime_conversion() -> None:
    """Verify conversion of UTC string and naive timestamp to MYT."""
    utc_str = "2024-01-10 02:00:00"
    myt_dt = to_myt_datetime(utc_str)
    assert myt_dt.hour == 10  # 02:00 UTC + 8 = 10:00 MYT
    assert myt_dt.day == 10

    # None fallback
    fallback_dt = to_myt_datetime(None)
    assert fallback_dt.tzinfo is not None


def test_to_myt_str_formatting() -> None:
    """Verify clean string formatting in MYT."""
    utc_str = "2024-01-10 04:30:00"
    result = to_myt_str(utc_str)
    assert result == "2024-01-10 12:30:00"

    assert to_myt_str(None) == "N/A"
    assert to_myt_str("") == "N/A"


def test_to_myt_series() -> None:
    """Verify series conversion to Asia/Kuala_Lumpur."""
    s = pd.Series(["2024-01-10 00:00:00", "2024-01-10 06:00:00"])
    myt_s = to_myt_series(s)
    assert str(myt_s.dt.tz) == MYT_TIMEZONE_STR
    assert myt_s.iloc[0].hour == 8
    assert myt_s.iloc[1].hour == 14


def test_is_ghost_hours_myt() -> None:
    """Verify Malaysian Ghost Hours detection (22:00 to 04:59 MYT)."""
    # 23:00 MYT (falls in ghost hours)
    assert is_ghost_hours_myt("2024-01-10 15:00:00") is True  # 15:00 UTC = 23:00 MYT
    # 03:00 MYT (falls in ghost hours)
    assert is_ghost_hours_myt("2024-01-10 19:00:00") is True  # 19:00 UTC = 03:00 MYT (+1 day)
    # 14:00 MYT (daytime)
    assert is_ghost_hours_myt("2024-01-10 06:00:00") is False  # 06:00 UTC = 14:00 MYT


def test_forensic_leaderboard(mock_telematics_service: TelematicsDataService) -> None:
    """Verify leaderboard generation and sorting."""
    lb = mock_telematics_service.get_forensic_leaderboard()
    assert not lb.empty
    assert "rank" in lb.columns
    assert "safety_score" in lb.columns
    assert "fuel_loss_liters" in lb.columns
    assert "critical_incidents" in lb.columns

    # Check vehicle 1_10_7 presence and fuel loss
    row_107 = lb[lb["vehicle_id"] == "1_10_7"]
    assert not row_107.empty
    assert row_107.iloc[0]["fuel_loss_liters"] == 5.0
    assert row_107.iloc[0]["critical_incidents"] >= 2


def test_dynamic_mcda_weight_recalculation(mock_telematics_service: TelematicsDataService) -> None:
    """Verify dynamic MCDA weight scaling recalculates scores live."""
    # Heavy fuel theft penalty
    heavy_fuel_weights = {"fuel_theft": 5.0, "moonlighting": 1.0, "speeding": 1.0, "harsh_driving": 1.0, "idling": 1.0}
    lb_heavy = mock_telematics_service.get_forensic_leaderboard(custom_weights=heavy_fuel_weights)

    # Light fuel theft penalty
    light_fuel_weights = {"fuel_theft": 1.0, "moonlighting": 1.0, "speeding": 1.0, "harsh_driving": 1.0, "idling": 1.0}
    lb_light = mock_telematics_service.get_forensic_leaderboard(custom_weights=light_fuel_weights)

    score_heavy = lb_heavy[lb_heavy["vehicle_id"] == "1_10_7"].iloc[0]["safety_score"]
    score_light = lb_light[lb_light["vehicle_id"] == "1_10_7"].iloc[0]["safety_score"]

    # Light penalty should yield higher or equal DSS score
    assert score_light >= score_heavy


def test_radar_profile_metrics(mock_telematics_service: TelematicsDataService) -> None:
    """Verify 5-axis explainable radar profile metrics."""
    radar = mock_telematics_service.get_radar_profile_metrics("1_10_7")
    assert "vehicle_scores" in radar
    assert "fleet_averages" in radar
    assert "report" in radar

    scores = radar["vehicle_scores"]
    assert "Fuel Integrity" in scores
    assert "Route Compliance" in scores
    assert "Kinematic Smoothness" in scores
    assert "Speed Adherence" in scores
    assert "Idling Discipline" in scores
    assert 0.0 <= scores["Fuel Integrity"] <= 100.0


def test_telegram_webhook_simulation(mock_telematics_service: TelematicsDataService) -> None:
    """Verify simulated Telegram webhook payload generation and audit logging."""
    receipt = mock_telematics_service.simulate_telegram_webhook(
        vehicle_id="1_10_7",
        incident_type="FUEL_THEFT",
        magnitude=-5.0,
        chat_id="-10099887766",
    )
    assert receipt["status"] == "DELIVERED"
    assert receipt["http_status"] == 200
    assert "payload" in receipt
    assert receipt["payload"]["chat_id"] == "-10099887766"
    assert "FUEL_THEFT" in receipt["payload"]["text"]

    # Verify action was logged to SQLite audit logs
    audit_logs = mock_telematics_service.get_audit_trail()
    assert not audit_logs.empty
    assert any(log["action"] == "DISPATCH_TELEGRAM_WEBHOOK" for _, log in audit_logs.iterrows())


def test_replay_telemetry_snapshot(mock_telematics_service: TelematicsDataService) -> None:
    """Verify historical replay telemetry snapshot queries."""
    # Before any pings
    snap_early = mock_telematics_service.get_replay_telemetry_snapshot(as_of_timestamp="2024-01-09 00:00:00")
    assert not snap_early.empty
    row = snap_early[snap_early["vehicle_id"] == "1_10_7"]
    assert row.iloc[0]["speed"] is None

    # After first ping
    snap_after = mock_telematics_service.get_replay_telemetry_snapshot(as_of_timestamp="2024-01-10 10:05:00")
    row_after = snap_after[snap_after["vehicle_id"] == "1_10_7"]
    assert row_after.iloc[0]["speed"] == 85.0


def test_telegram_notification_service() -> None:
    """Verify TelegramNotificationService configuration, formatting, and dispatch."""
    from src.services.telegram_service import TelegramNotificationService

    service = TelegramNotificationService(bot_token="test_mock_token_12345:ABCDE", chat_id="-100998877")
    assert service.is_configured is True
    assert service.get_masked_token() == "test...BCDE"

    # Test payload formatting: zero emojis, proper headers, MYT timestamp
    payload_str = service.format_incident_payload(
        vehicle_id="1_10_7",
        incident_type="FUEL_THEFT",
        magnitude=-5.0,
        timestamp="2024-01-10 10:15:00",
    )
    assert "[CRITICAL TELEMATICS SECURITY ALERT]" in payload_str
    assert "Vehicle ID: 1_10_7" in payload_str
    assert "Timestamp (MYT): 2024-01-10 18:15:00" in payload_str  # 10:15 UTC + 8 = 18:15 MYT
    assert "Decision Engine" in payload_str

    # Test test alert dispatch
    test_res = service.send_test_alert(chat_id="-100998877")
    assert test_res["status"] in ("DELIVERED", "API_ERROR", "NETWORK_FALLBACK")
    assert test_res["chat_id"] == "-100998877"

    # Test trigger anomaly alert dispatch
    alert_res = service.trigger_anomaly_alert(
        vehicle_id="1_10_7",
        incident_type="FUEL_THEFT",
        magnitude=-5.0,
    )
    assert alert_res["status"] in ("DELIVERED", "API_ERROR", "NETWORK_FALLBACK")


def test_mode_b_calendar_date_and_shift_filtering(mock_telematics_service: TelematicsDataService) -> None:
    """Verify Mode B independent historical date and shift filtering."""
    # Dates query
    dates = mock_telematics_service.get_available_historical_dates()
    assert "2024-01-10" in dates

    # Leaderboard on matching date
    lb_match = mock_telematics_service.get_forensic_leaderboard(
        target_date="2024-01-10",
        shift_window="Full 24-Hour Shift",
    )
    assert not lb_match.empty
    assert "1_10_7" in lb_match["vehicle_id"].values

    # Leaderboard on non-existent date
    lb_empty = mock_telematics_service.get_forensic_leaderboard(
        target_date="2024-05-20",
        shift_window="Full 24-Hour Shift",
    )
    assert lb_empty.empty

    # Fuel loss records on matching date
    fuel_records = mock_telematics_service.get_fuel_loss_records(
        target_date="2024-01-10",
        shift_window="Full 24-Hour Shift",
    )
    assert not fuel_records.empty
    assert fuel_records.iloc[0]["vehicle_id"] == "1_10_7"

    # Vehicle telemetry on matching date
    telemetry = mock_telematics_service.get_vehicle_telemetry(
        vehicle_id="1_10_7",
        target_date="2024-01-10",
        shift_window="Full 24-Hour Shift",
    )
    assert len(telemetry) == 2


def test_anomaly_resolution_flow(mock_telematics_service: TelematicsDataService) -> None:
    """Verify operational acknowledgment and resolution of active unread anomalies."""
    anoms = mock_telematics_service.get_anomalies(vehicle_id="1_10_7", status="OPEN")
    assert not anoms.empty
    first_id = int(anoms.iloc[0]["id"])

    # Resolve anomaly
    success, msg = mock_telematics_service.resolve_anomaly(
        anomaly_id=first_id,
        resolution_notes="Driver verified; maintenance inspection passed.",
        user_id="lead_dispatcher",
    )
    assert success is True

    # Verify status changed in database
    updated_anoms = mock_telematics_service.get_anomalies(vehicle_id="1_10_7", status="OPEN")
    assert first_id not in updated_anoms["id"].values
