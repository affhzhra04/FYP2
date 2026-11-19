"""Incident priority rules and supervisor alerts."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src import dash_core as C

PRIORITIES = ["High", "Medium", "Low"]


def priority(ev: pd.DataFrame, speed_limit: float = 90.0) -> pd.Series:
    """High / Medium / Low per event (vectorised).

    High: any fuel loss tied to an off-route or unknown stop, any stationary fuel loss, speeding 15 km/h or more
    over the limit, a stop of 2 h or longer at an unknown place.
    Low: idling. Everything else is Medium.
    """
    k = ev["kind"]
    m = ev["magnitude"].astype(float)
    d = ev["duration_s"].astype(float)
    pr = pd.Series("Medium", index=ev.index)
    pr[k == "idling"] = "Low"
    high = (k.isin(["fuel_theft", "fuel_off_route"])
            | ((k == "speeding") & (m >= speed_limit + 15))
            | ((k == "unauthorized_stop") & (d >= 7200)))
    pr[high] = "High"
    return pr


def alert_text(vehicle: str, event: str, detail: str, when: str, priority_: str = "High") -> str:
    """Plain-text message (no Markdown special characters, so the Telegram API accepts it)."""
    def clean(s: str) -> str:
        return str(s).replace("_", "-").replace("*", "").replace("`", "").replace("[", "(").replace("]", ")")
    return (f"DRIVER RISK ALERT ({priority_} priority)\n"
            f"Vehicle: {clean(vehicle)}\n"
            f"Event: {clean(event)}\n"
            f"Detail: {clean(detail)}\n"
            f"Time (MYT): {when}\n"
            "Action: please review in the Driver Risk Monitor.")


def dispatch(text: str) -> Tuple[str, str, str]:
    """Send via the existing Telegram gateway. Returns (mode, status, note).

    Without a bot token in .env the gateway runs in simulation mode, which is reported as such.
    """
    try:
        from src.services.telegram_service import TelegramNotificationService
        r: Dict = TelegramNotificationService().dispatch_alert(text)
        return str(r.get("mode", "SIMULATED")), str(r.get("status", "UNKNOWN")), str(
            r.get("notice") or r.get("error_details") or "")
    except Exception as exc:                                   # missing deps / bad config
        return "SIMULATED", "NOT_SENT", f"Telegram gateway unavailable: {exc}"
