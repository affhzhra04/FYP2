"""Unit tests for UI components and design system utilities."""

from __future__ import annotations
from unittest.mock import patch
import pytest

import src.components as components_pkg
from src.components.ui import (
    ENTERPRISE_CSS,
    get_freshness_badge_html,
    get_risk_tier_badge_html,
    inject_custom_css,
    render_empty_state,
    render_metric_card,
    render_notice_banner,
    render_page_header,
)


def test_package_exports_consistency() -> None:
    """Ensure all required public symbols are exported in both ui.py and components.__init__."""
    expected_exports = [
        "ENTERPRISE_CSS",
        "inject_custom_css",
        "render_page_header",
        "render_metric_card",
        "render_notice_banner",
        "render_empty_state",
        "get_freshness_badge_html",
        "get_risk_tier_badge_html",
    ]
    for symbol in expected_exports:
        assert hasattr(components_pkg, symbol), f"Symbol {symbol} missing from src.components"
        assert symbol in components_pkg.__all__, f"Symbol {symbol} missing from src.components.__all__"


def test_render_notice_banner_markup() -> None:
    """Verify render_notice_banner generates expected enterprise HTML markup."""
    with patch("streamlit.markdown") as mock_markdown:
        render_notice_banner(
            message="Telemetry stream connected.",
            banner_type="info",
            title="Operational Status",
            badge="LIVE",
        )
        assert mock_markdown.called
        rendered_html = mock_markdown.call_args[0][0]
        assert "fleet-banner-info" in rendered_html
        assert "Telemetry stream connected." in rendered_html
        assert "Operational Status" in rendered_html
        assert "LIVE" in rendered_html


@pytest.mark.parametrize(
    "banner_type,expected_class",
    [
        ("info", "fleet-banner-info"),
        ("warning", "fleet-banner-warning"),
        ("error", "fleet-banner-error"),
        ("danger", "fleet-banner-error"),
        ("success", "fleet-banner-success"),
    ],
)
def test_render_notice_banner_types(banner_type: str, expected_class: str) -> None:
    """Test banner type class mapping."""
    with patch("streamlit.markdown") as mock_markdown:
        render_notice_banner(message="Test message", banner_type=banner_type)
        rendered_html = mock_markdown.call_args[0][0]
        assert expected_class in rendered_html


def test_render_empty_state_robustness() -> None:
    """Test render_empty_state with custom title, description, and legacy icon parameter."""
    with patch("streamlit.markdown") as mock_markdown:
        render_empty_state(
            title="No Results Found",
            description="Adjust your search filters.",
            icon="search",
        )
        assert mock_markdown.called
        rendered_html = mock_markdown.call_args[0][0]
        assert "No Results Found" in rendered_html
        assert "Adjust your search filters." in rendered_html


def test_freshness_and_tier_badge_html() -> None:
    """Verify badge HTML generation functions."""
    active_html = get_freshness_badge_html("Active")
    assert "badge-active" in active_html

    stale_html = get_freshness_badge_html("Stale")
    assert "badge-stale" in stale_html

    crit_html = get_risk_tier_badge_html("Critical")
    assert "tier-critical" in crit_html

    exc_html = get_risk_tier_badge_html("Excellent")
    assert "tier-excellent" in exc_html


def test_render_page_header_clean_html() -> None:
    """Verify render_page_header generates clean HTML without newlines or indentation that triggers CommonMark code blocks."""
    with patch("streamlit.markdown") as mock_markdown:
        render_page_header(
            title="Fleet Overview",
            description="Monitoring dashboard",
            meta_badges=[("System", "Operational"), ("Window", "24h")],
        )
        assert mock_markdown.called
        rendered_html = mock_markdown.call_args[0][0]
        # Must contain HTML elements
        assert "fleet-page-header" in rendered_html
        assert "fleet-meta-chip" in rendered_html
        assert "Operational" in rendered_html
        # Must NOT contain newlines or 4-space indentations
        assert "\n" not in rendered_html
        assert "    " not in rendered_html


def test_render_metric_card_clean_html() -> None:
    """Verify render_metric_card generates clean HTML without leading indentation."""
    with patch("streamlit.markdown") as mock_markdown:
        render_metric_card(
            label="Active Units",
            value="12 / 14",
            subtitle="Reporting",
            delta="12 Online",
            delta_color="normal",
        )
        assert mock_markdown.called
        rendered_html = mock_markdown.call_args[0][0]
        assert "fleet-metric-card" in rendered_html
        assert "12 / 14" in rendered_html
        assert "fleet-metric-delta-pos" in rendered_html
        assert "\n" not in rendered_html
        assert "    " not in rendered_html

