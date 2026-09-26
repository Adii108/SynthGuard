"""
Stitch UI package for SynthGuard.
"""
from synthguard.ui.styles import STITCH_CSS
from synthguard.ui.components import render_stitch_header, render_kpi_bar
from synthguard.ui.views_overview import render_overview_view
from synthguard.ui.views_dataset import render_dataset_view
from synthguard.ui.views_generate import render_generate_view
from synthguard.ui.views_evaluate import render_evaluate_view
from synthguard.ui.views_privacy import render_privacy_view
from synthguard.ui.views_report import render_report_view

__all__ = [
    "STITCH_CSS",
    "render_stitch_header",
    "render_kpi_bar",
    "render_overview_view",
    "render_dataset_view",
    "render_generate_view",
    "render_evaluate_view",
    "render_privacy_view",
    "render_report_view",
]
