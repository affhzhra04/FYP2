"""Malaysia Standard Time (MYT, UTC+8) Timezone Utility Module.

Provides deterministic datetime transformations, timezone localization,
and string formatting enforcing Asia/Kuala_Lumpur across telematics pipelines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

# Canonical Malaysia Standard Time (UTC+8) zone identifier
MYT_TIMEZONE_STR = "Asia/Kuala_Lumpur"
MYT_TZ = ZoneInfo(MYT_TIMEZONE_STR)
UTC_TZ = timezone.utc


def now_myt() -> datetime:
    """Return the current system time in Malaysia Standard Time (MYT, UTC+8)."""
    return datetime.now(MYT_TZ)


def to_myt_datetime(val: Any) -> datetime:
    """Convert any datetime, pandas Timestamp, or ISO string to MYT (Asia/Kuala_Lumpur).

    If the input is timezone-naive, it is assumed to represent UTC and localized,
    then converted to Asia/Kuala_Lumpur.
    """
    if pd.isna(val) or val is None:
        return now_myt()

    if isinstance(val, str):
        # Parse ISO or standard timestamp string
        dt = pd.to_datetime(val)
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(MYT_TIMEZONE_STR).to_pydatetime()

    if isinstance(val, pd.Timestamp):
        if val.tz is None:
            val = val.tz_localize("UTC")
        return val.tz_convert(MYT_TIMEZONE_STR).to_pydatetime()

    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=UTC_TZ)
        return val.astimezone(MYT_TZ)

    # Fallback to current time in MYT
    return now_myt()


def to_myt_str(val: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a timestamp into a clean Malaysia Standard Time (MYT) string."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return "N/A"
    try:
        dt = to_myt_datetime(val)
        return dt.strftime(fmt)
    except Exception:
        return str(val)


def to_myt_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series of timestamps to timezone-aware MYT (Asia/Kuala_Lumpur)."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        # If naive, assume UTC and convert to MYT
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(MYT_TIMEZONE_STR)


def is_ghost_hours_myt(val: Any, ghost_start: int = 22, ghost_end: int = 5) -> bool:
    """Evaluate whether a timestamp falls in Malaysian Ghost Hours (22:00 to 04:59 MYT)."""
    try:
        dt = to_myt_datetime(val)
        hr = dt.hour
        return (hr >= ghost_start) or (hr < ghost_end)
    except Exception:
        return False
