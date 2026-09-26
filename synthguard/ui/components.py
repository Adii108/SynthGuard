"""
Reusable Stitch UI Components for SynthGuard.
"""
import streamlit as st
from typing import Optional

def render_stitch_header(active_page: str, dataset_name: Optional[str] = None):
    """Renders the Stitch top header with dataset status and 5-phase step tracker."""
    ds_display = dataset_name if dataset_name and dataset_name != "None" else "No Dataset Selected"
    status_class = "stitch-badge-ready" if dataset_name and dataset_name != "None" else ""
    status_label = "Ready" if dataset_name and dataset_name != "None" else "Awaiting Data"

    steps = [
        ("Overview", "Overview"),
        ("1. Dataset", "Dataset"),
        ("2. Generate", "Generate"),
        ("3. Evaluate", "Evaluate"),
        ("4. Privacy", "Privacy"),
        ("5. Report", "Report"),
    ]

    step_items_html = []
    for idx, (label, key) in enumerate(steps):
        is_active = (key == active_page)
        active_cls = "active" if is_active else ""
        num_label = str(idx) if idx > 0 else "•"
        step_items_html.append(
            f'<span class="stitch-step-item {active_cls}">'
            f'<span class="stitch-step-num">{num_label}</span>{label}'
            f'</span>'
        )

    stepper_html = '<span class="stitch-step-arrow">→</span>'.join(step_items_html)

    header_html = f"""
    <div class="stitch-header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 32px; height: 32px; border-radius: 6px; background: #06271F; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-weight: 700; font-size: 16px;">
                    🛡️
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-family: 'Source Serif 4', serif; font-size: 18px; font-weight: 600; color: #06271F;">SynthGuard</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; background: #E5E9E4; color: #414845; padding: 2px 6px; border-radius: 4px;">{ds_display}</span>
                    </div>
                    <span style="font-size: 11px; color: #717975;">Empirical Synthesis & Privacy Intelligence</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
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
    """Renders a responsive high-density KPI metrics bar."""
    cols = st.columns(len(kpi_data))
    for col, (label, val, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="stitch-card" style="padding: 12px 16px; text-align: center; margin-bottom: 8px;">
                <div style="font-size: 11px; color: #717975; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">{label}</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 600; color: #06271F; margin: 4px 0;">{val}</div>
                <div style="font-size: 11px; color: #414845;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
