"""
Refined Stitch UI Components for SynthGuard.
"""
import streamlit as st
from typing import Optional, List, Tuple

def render_stitch_header(active_page: str, dataset_name: Optional[str] = None):
    """Renders the Stitch top header with dataset status and 5-phase step tracker."""
    has_data = dataset_name and dataset_name != "None"
    ds_display = dataset_name if has_data else "No Dataset Ingested"
    status_class = "stitch-badge-ready" if has_data else "stitch-badge-awaiting"
    status_label = "Active Dataset" if has_data else "Awaiting Data"

    steps: List[Tuple[str, str, int]] = [
        ("Overview", "Overview", 1),
        ("Dataset", "Dataset", 2),
        ("Generate", "Generate", 3),
        ("Evaluate", "Evaluate", 4),
        ("Privacy", "Privacy", 5),
        ("Report", "Report", 6),
    ]

    step_items_html = []
    for label, key, num in steps:
        is_active = (key == active_page)
        active_cls = "active" if is_active else ""
        step_items_html.append(
            f'<span class="stitch-step-item {active_cls}">'
            f'<span class="stitch-step-num">{num:02d}</span>{label}'
            f'</span>'
        )

    stepper_html = '<span class="stitch-step-arrow">→</span>'.join(step_items_html)

    header_html = f"""
    <div class="stitch-header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 36px; height: 36px; border-radius: 8px; background: #0A2E24; display: flex; align-items: center; justify-content: center; font-size: 18px; box-shadow: 0 2px 4px rgba(10, 46, 36, 0.15);">
                    🛡️
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span style="font-family: 'Source Serif 4', Georgia, serif; font-size: 18px; font-weight: 700; color: #06271F; letter-spacing: -0.01em;">SynthGuard</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; background: #E5EDE7; color: #0A2E24; padding: 2px 8px; border-radius: 4px; border: 1px solid #C8D6CC;">{ds_display}</span>
                    </div>
                    <span style="font-size: 11px; font-weight: 500; color: #4A6357;">Empirical Tabular Synthesis & Privacy Intelligence</span>
                </div>
            </div>
            <div>
                <span class="{status_class}">{status_label}</span>
            </div>
        </div>
        <div class="stitch-stepper">
            {stepper_html}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_kpi_bar(kpi_data: list):
    """Renders a responsive high-density KPI metrics bar with high contrast."""
    cols = st.columns(len(kpi_data))
    for col, (label, val, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="stitch-card" style="padding: 14px 16px; text-align: center; margin-bottom: 10px;">
                <div style="font-size: 11px; color: #4A6357; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">{label}</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #06271F; margin: 4px 0;">{val}</div>
                <div style="font-size: 12px; color: #2E453B; font-weight: 500;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
