"""Views package for FYP2 Streamlit Dashboard."""
from src.views.realtime_operations import render_realtime_operations_view
from src.views.forensic_audit import render_forensic_audit_view
from src.views.fleet_overview import render_fleet_overview_page
from src.views.signal_inspector import render_signal_inspector_page
from src.views.risk_trends import render_risk_trends_page
from src.views.explainable_scoring import render_explainable_scoring_page
from src.views.fleet_config import render_fleet_config_page

__all__ = [
    "render_realtime_operations_view",
    "render_forensic_audit_view",
    "render_fleet_overview_page",
    "render_signal_inspector_page",
    "render_risk_trends_page",
    "render_explainable_scoring_page",
    "render_fleet_config_page",
]
