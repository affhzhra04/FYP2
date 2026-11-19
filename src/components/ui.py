"""Enterprise UI Design System & Component Library for Fleet Operation Dashboard.

Implements a refined, high-density light-theme SaaS design system adhering to 4px spacing,
deliberate typography scale, restrained operational color tokens, and robust B2B UI components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import streamlit as st


# ==============================================================================
# Enterprise CSS Design System (Refined Light Theme)
# ==============================================================================

ENTERPRISE_CSS = """
<style>
/* Base Reset & Typography Scale */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0f172a;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

/* App Canvas */
.stApp {
    background-color: #f8fafc;
}

/* Header toolbar compacting */
header[data-testid="stHeader"] {
    height: 1.5rem !important;
    background: transparent !important;
}

/* Main Container Spacing - Compact 4px scale */
.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    max-width: 100% !important;
}

/* Compact layout gaps */
div[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
div[data-testid="stHorizontalBlock"] {
    gap: 0.65rem !important;
}

/* Header & Typography Controls */
h1 {
    font-size: 1.25rem !important; /* 20px */
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.15rem !important;
    padding-top: 0 !important;
    line-height: 1.2 !important;
}

h2 {
    font-size: 1.0625rem !important; /* 17px */
    font-weight: 600 !important;
    color: #0f172a !important;
    letter-spacing: -0.01em !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.35rem !important;
}

h3 {
    font-size: 0.9375rem !important; /* 15px */
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-top: 0.35rem !important;
    margin-bottom: 0.25rem !important;
}

/* Sidebar Styling - Authoritative Navy Enterprise Surface */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
    box-shadow: none !important;
    min-width: 265px !important;
    max-width: 265px !important;
    width: 265px !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}

/* Sidebar Branding Header */
.sidebar-brand-box {
    padding: 0 0.25rem 0.75rem 0.25rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 0.75rem;
}
.sidebar-brand-title {
    font-size: 0.9375rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-brand-sub {
    font-size: 0.6875rem;
    color: #94a3b8;
    margin-top: 2px;
}
.sidebar-live-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.625rem;
    font-weight: 700;
    color: #22c55e;
    background-color: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    padding: 1px 5px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Sidebar Navigation Items */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 6px 10px !important;
    border-radius: 5px !important;
    font-size: 0.8125rem !important; /* 13px */
    font-weight: 500 !important;
    color: #94a3b8 !important;
    cursor: pointer !important;
    transition: all 0.12s ease-in-out !important;
    border: 1px solid transparent !important;
    display: flex !important;
    align-items: center !important;
    margin-bottom: 1px !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: rgba(255, 255, 255, 0.06) !important;
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background-color: #1e293b !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-left: 3px solid #3b82f6 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* Sidebar Sliders & Controls Styling - High Visibility against Dark Navy */
section[data-testid="stSidebar"] [data-testid="stSlider"] label,
section[data-testid="stSidebar"] [data-testid="stSelectSlider"] label,
section[data-testid="stSidebar"] .stSlider label {
    color: #f1f5f9 !important;
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.25rem !important;
}

/* Track container */
section[data-testid="stSidebar"] div[data-baseweb="slider"] {
    background-color: transparent !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}

/* Background Unfilled Track */
section[data-testid="stSidebar"] div[data-baseweb="slider"] > div {
    overflow: visible !important;
    background-color: #475569 !important; /* Visible slate-500 */
    height: 8px !important;
    border-radius: 4px !important;
}

/* Filled Active Track */
section[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div:first-child {
    background-color: #3b82f6 !important; /* High-contrast enterprise blue */
    height: 8px !important;
    border-radius: 4px !important;
}

/* Slider Thumb Handle */
section[data-testid="stSidebar"] div[role="slider"],
section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: #ffffff !important; /* Bright white center */
    border: 3px solid #3b82f6 !important; /* Bold blue border */
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.4), 0 2px 6px rgba(0, 0, 0, 0.6) !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    cursor: grab !important;
    z-index: 10 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}

section[data-testid="stSidebar"] div[role="slider"]:hover {
    transform: scale(1.15) !important;
    box-shadow: 0 0 0 5px rgba(59, 130, 246, 0.6) !important;
}

/* Tick bar & tick marks */
section[data-testid="stSidebar"] div[data-testid="stSliderTickBar"] {
    margin-top: 6px !important;
}

section[data-testid="stSidebar"] div[data-testid="stSliderTickBar"] span,
section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stSlider"] span,
section[data-testid="stSidebar"] [data-testid="stSelectSlider"] span,
section[data-testid="stSidebar"] div[data-baseweb="slider"] span {
    color: #f8fafc !important; /* Crisp high-contrast white */
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

/* Current value display */
section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stSelectSlider"] div[data-testid="stThumbValue"] {
    color: #ffffff !important;
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

/* Sidebar Operations Card */
.sidebar-ops-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 8px 10px;
    margin-top: 1rem;
    font-size: 0.6875rem;
    color: #94a3b8;
}
.sidebar-ops-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}
.sidebar-ops-val {
    color: #f1f5f9;
    font-weight: 600;
}

/* Compact Executive Metric Card */
.fleet-metric-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    min-height: 72px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.15s ease;
}

.fleet-metric-card:hover {
    border-color: #cbd5e1;
}

.fleet-metric-label {
    font-size: 0.6875rem; /* 11px */
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.fleet-metric-value-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 6px;
}

.fleet-metric-value {
    font-size: 1.375rem; /* 22px */
    font-weight: 700;
    line-height: 1.15;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.fleet-metric-delta {
    font-size: 0.6875rem; /* 11px */
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 3px;
    white-space: nowrap;
}
.fleet-metric-delta-pos {
    background-color: #f0fdf4;
    color: #16a34a;
    border: 1px solid #bbf7d0;
}
.fleet-metric-delta-neg {
    background-color: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

.fleet-metric-sub {
    font-size: 0.6875rem; /* 11px */
    color: #94a3b8;
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Compact Page Header Container */
.fleet-page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    padding-bottom: 8px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 8px;
}

.fleet-page-title-group {
    flex: 1;
    min-width: 240px;
}

.fleet-page-title-group h1 {
    margin: 0 !important;
}

.fleet-page-desc {
    font-size: 0.75rem; /* 12px */
    color: #64748b;
    margin: 2px 0 0 0;
    line-height: 1.3;
}

.fleet-header-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.fleet-meta-chip {
    font-size: 0.6875rem; /* 11px */
    font-weight: 500;
    color: #475569;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 3px 7px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    line-height: 1.2;
}

.fleet-meta-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #16a34a;
    display: inline-block;
}

/* Status Badges */
.badge-active {
    background-color: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
    display: inline-block;
}
.badge-recent {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
    display: inline-block;
}
.badge-stale {
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
    display: inline-block;
}
.badge-none {
    background-color: #f8fafc;
    color: #64748b;
    border: 1px solid #e2e8f0;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 500;
    display: inline-block;
}

/* Risk Tier Badges */
.tier-excellent {
    background-color: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
}
.tier-low {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
}
.tier-moderate {
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
}
.tier-high {
    background-color: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
}
.tier-critical {
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.6875rem;
    font-weight: 600;
}

/* Empty States */
.fleet-empty-state {
    text-align: center;
    padding: 1.5rem 1rem;
    background-color: #ffffff;
    border: 1px dashed #cbd5e1;
    border-radius: 6px;
    margin: 8px 0;
}
.fleet-empty-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 3px;
}
.fleet-empty-desc {
    font-size: 0.75rem;
    color: #64748b;
    max-width: 440px;
    margin: 0 auto;
}

/* Filter Strip Container */
.fleet-filter-bar {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 6px 10px;
    margin-bottom: 8px;
}

/* Form Controls & Inputs Overrides */
div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
    border-radius: 4px !important;
}
div[data-baseweb="select"] {
    border-radius: 4px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 4px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    padding: 5px 12px !important;
    border: 1px solid #cbd5e1 !important;
}
.stButton > button[kind="primary"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1e293b !important;
    border-color: #1e293b !important;
}

/* Notice Banners */
.fleet-banner {
    padding: 8px 12px;
    border-radius: 5px;
    font-size: 0.75rem; /* 12px */
    line-height: 1.4;
    margin: 6px 0 10px 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.fleet-banner-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 1px;
}
.fleet-banner-title {
    font-weight: 600;
    font-size: 0.75rem;
}
.fleet-banner-badge {
    font-size: 0.625rem; /* 10px */
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 1px 4px;
    border-radius: 3px;
}
.fleet-banner-msg {
    margin: 0;
}

/* Banner Variations */
.fleet-banner-info {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb;
    color: #1e3a8a;
}
.fleet-banner-info .fleet-banner-badge {
    background-color: #dbeafe;
    color: #1d4ed8;
}

.fleet-banner-warning {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #d97706;
    color: #78350f;
}
.fleet-banner-warning .fleet-banner-badge {
    background-color: #fef3c7;
    color: #b45309;
}

.fleet-banner-error, .fleet-banner-danger {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 3px solid #dc2626;
    color: #7f1d1d;
}
.fleet-banner-error .fleet-banner-badge, .fleet-banner-danger .fleet-banner-badge {
    background-color: #fee2e2;
    color: #b91c1c;
}

.fleet-banner-success {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid #16a34a;
    color: #14532d;
}
.fleet-banner-success .fleet-banner-badge {
    background-color: #dcfce7;
    color: #15803d;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject enterprise CSS styling into the Streamlit app."""
    st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


# ==============================================================================
# Reusable Enterprise Layout & Component Helpers
# ==============================================================================

def _clean_html(raw_html: str) -> str:
    """Strip leading/trailing spaces from each line and join without blank lines to prevent CommonMark indented-code parsing."""
    return "".join(line.strip() for line in raw_html.strip().splitlines() if line.strip())


def render_page_header(
    title: str,
    description: str,
    meta_badges: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Render a compact, executive page header with operational metadata chips.

    Parameters
    ----------
    title : str
        Primary page title.
    description : str
        Concise operational context description.
    meta_badges : Optional[List[Tuple[str, str]]]
        List of (label, value) metadata chips to display right-aligned.
    """
    chips_html = ""
    if meta_badges:
        for label, val in meta_badges:
            chips_html += (
                f'<div class="fleet-meta-chip">'
                f'<span class="fleet-meta-dot"></span>'
                f'<span><strong>{label}:</strong> {val}</span>'
                f'</div>'
            )

    html = (
        f'<div class="fleet-page-header">'
        f'<div class="fleet-page-title-group">'
        f'<h1>{title}</h1>'
        f'<p class="fleet-page-desc">{description}</p>'
        f'</div>'
        f'<div class="fleet-header-meta">{chips_html}</div>'
        f'</div>'
    )
    st.markdown(_clean_html(html), unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: Any,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal",
) -> None:
    """Render an executive KPI card with consistent dimensions and restrained borders.

    Parameters
    ----------
    label : str
        Compact metric header.
    value : Any
        Primary numeric value.
    subtitle : Optional[str]
        Secondary supporting context.
    delta : Optional[str]
        Operational status delta indicator.
    delta_color : str
        'normal' (green) or 'inverse' (red).
    """
    delta_html = ""
    if delta:
        cls = "fleet-metric-delta-pos" if delta_color == "normal" else "fleet-metric-delta-neg"
        delta_html = f'<span class="fleet-metric-delta {cls}">{delta}</span>'

    sub_html = f'<div class="fleet-metric-sub">{subtitle}</div>' if subtitle else ""

    html = (
        f'<div class="fleet-metric-card">'
        f'<div>'
        f'<div class="fleet-metric-label">{label}</div>'
        f'<div class="fleet-metric-value-row">'
        f'<span class="fleet-metric-value">{value}</span>'
        f'{delta_html}'
        f'</div>'
        f'</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(_clean_html(html), unsafe_allow_html=True)


def render_notice_banner(
    message: str,
    banner_type: str = "info",
    title: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Render a clean, enterprise-grade operational notice banner.

    Parameters
    ----------
    message : str
        Primary message text or explanation.
    banner_type : str, default "info"
        Type of banner: 'info', 'warning', 'error', 'danger', 'success'.
    title : Optional[str], default None
        Optional bold title displayed within the banner.
    badge : Optional[str], default None
        Optional short category or status tag (e.g. 'LIVE', 'OFFLINE').
    """
    b_type = (banner_type or "info").lower().strip()
    if b_type in ("error", "danger", "critical"):
        cls = "fleet-banner-error"
    elif b_type in ("warning", "warn", "alert"):
        cls = "fleet-banner-warning"
    elif b_type in ("success", "ok"):
        cls = "fleet-banner-success"
    else:
        cls = "fleet-banner-info"

    header_parts = []
    if badge:
        header_parts.append(f'<span class="fleet-banner-badge">{badge}</span>')
    if title:
        header_parts.append(f'<span class="fleet-banner-title">{title}</span>')

    header_html = f'<div class="fleet-banner-row">{" ".join(header_parts)}</div>' if header_parts else ""

    html = (
        f'<div class="fleet-banner {cls}">'
        f'{header_html}'
        f'<div class="fleet-banner-msg">{message}</div>'
        f'</div>'
    )
    st.markdown(_clean_html(html), unsafe_allow_html=True)


def render_empty_state(
    title: str = "No Records Found",
    description: str = "The selected operational query returned zero matching records from the database.",
    icon: Optional[str] = None,
) -> None:
    """Render a clean, minimal enterprise empty state container.

    Parameters
    ----------
    title : str
        Primary title text for the empty state.
    description : str
        Detailed explanation or suggestion for resolution.
    icon : Optional[str]
        Optional icon identifier (ignored in refined light enterprise design).
    """
    _ = icon
    html = (
        f'<div class="fleet-empty-state">'
        f'<div class="fleet-empty-title">{title}</div>'
        f'<div class="fleet-empty-desc">{description}</div>'
        f'</div>'
    )
    st.markdown(_clean_html(html), unsafe_allow_html=True)


def get_freshness_badge_html(status: str) -> str:
    """Return compact HTML badge for connectivity status."""
    s = str(status).lower()
    if "active" in s:
        return '<span class="badge-active">Active</span>'
    elif "recent" in s:
        return '<span class="badge-recent">Recent</span>'
    elif "stale" in s:
        return '<span class="badge-stale">Stale</span>'
    return '<span class="badge-none">No Telemetry</span>'


def get_risk_tier_badge_html(tier: str) -> str:
    """Return compact HTML badge for risk classification tier."""
    t = str(tier).lower()
    if "excellent" in t:
        return '<span class="tier-excellent">Excellent</span>'
    elif "low" in t:
        return '<span class="tier-low">Low Risk</span>'
    elif "moderate" in t:
        return '<span class="tier-moderate">Moderate</span>'
    elif "high" in t:
        return '<span class="tier-high">High Risk</span>'
    elif "critical" in t:
        return '<span class="tier-critical">Critical</span>'
    return f'<span>{tier}</span>'


__all__ = [
    "ENTERPRISE_CSS",
    "inject_custom_css",
    "render_page_header",
    "render_metric_card",
    "render_notice_banner",
    "render_empty_state",
    "get_freshness_badge_html",
    "get_risk_tier_badge_html",
]


