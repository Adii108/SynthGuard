"""
SynthGuard: GenAI Platform for Synthetic Tabular Data Generation, Evaluation & Privacy Risk Analysis.
Integrated with Stitch Design System: 'Calm Editorial Intelligence' (Refined High-Contrast Theme).
"""
import synthguard.utils.numpy_compat  # Applies NumPy 2.x compatibility patches
import streamlit as st
import pandas as pd
import numpy as np

from synthguard.ui import (
    STITCH_CSS,
    render_stitch_header,
    render_overview_view,
    render_dataset_view,
    render_generate_view,
    render_evaluate_view,
    render_privacy_view,
    render_report_view
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SynthGuard - Synthetic Data & Privacy Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Refined Stitch Design System CSS
st.markdown(STITCH_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "df_real" not in st.session_state:
    st.session_state.df_real = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "None"
if "profile" not in st.session_state:
    st.session_state.profile = None
if "gen_config" not in st.session_state:
    st.session_state.gen_config = None
if "df_synth" not in st.session_state:
    st.session_state.df_synth = None
if "evaluation_summary" not in st.session_state:
    st.session_state.evaluation_summary = None
if "current_nav" not in st.session_state:
    st.session_state.current_nav = "Overview"

# ---------------------------------------------------------
# Sidebar Navigation (High-Contrast Redesigned Hierarchy)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
        <div style="width: 32px; height: 32px; border-radius: 6px; background: #0A2E24; display: flex; align-items: center; justify-content: center; font-size: 16px;">
            🛡️
        </div>
        <div>
            <span style="font-family: 'Source Serif 4', Georgia, serif; font-size: 19px; font-weight: 700; color: #06271F; letter-spacing: -0.01em;">SynthGuard</span>
        </div>
    </div>
    <div style="font-size: 12px; font-weight: 500; color: #3A5246; margin-bottom: 16px;">Empirical Tabular Synthesis</div>
    """, unsafe_allow_html=True)

    nav_options = [
        "01  Overview",
        "02  Dataset",
        "03  Generate",
        "04  Evaluate",
        "05  Privacy",
        "06  Report"
    ]
    
    nav_map = {
        "01  Overview": "Overview",
        "02  Dataset": "Dataset",
        "03  Generate": "Generate",
        "04  Evaluate": "Evaluate",
        "05  Privacy": "Privacy",
        "06  Report": "Report"
    }

    default_idx = 0
    for idx, (label, val) in enumerate(nav_map.items()):
        if val == st.session_state.current_nav:
            default_idx = idx
            break

    selected_label = st.radio("WORKFLOW", nav_options, index=default_idx)
    st.session_state.current_nav = nav_map[selected_label]

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; font-weight: 700; color: #3A5246; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>Active Dataset</div>", unsafe_allow_html=True)
    
    if st.session_state.df_real is not None:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #D4DFD7; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
            <div style="font-weight: 600; font-size: 13px; color: #06271F; margin-bottom: 4px;">{st.session_state.dataset_name}</div>
            <div style="font-size: 12px; color: #2E453B; font-family: 'JetBrains Mono', monospace;">📊 {len(st.session_state.df_real):,} rows</div>
            <div style="font-size: 12px; color: #2E453B; font-family: 'JetBrains Mono', monospace;">📐 {len(st.session_state.df_real.columns)} columns</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #D4DFD7; border-radius: 8px; padding: 12px; font-size: 12px; color: #4A6357;">
            No dataset ingested yet.
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.df_synth is not None:
        st.markdown(f"""
        <div style="background: #D1EAD9; border: 1px solid #B5C8BA; border-radius: 8px; padding: 10px 12px; font-size: 12px; font-weight: 600; color: #083D2A; margin-top: 8px;">
            🧪 Generated: {len(st.session_state.df_synth):,} rows
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.caption("SynthGuard v1.0.0 • Editorial Data Science UI")

# ---------------------------------------------------------
# Top Stitch Header
# ---------------------------------------------------------
render_stitch_header(st.session_state.current_nav, st.session_state.dataset_name)

# ---------------------------------------------------------
# Main View Routing
# ---------------------------------------------------------
if st.session_state.current_nav == "Overview":
    render_overview_view()
elif st.session_state.current_nav == "Dataset":
    render_dataset_view()
elif st.session_state.current_nav == "Generate":
    render_generate_view()
elif st.session_state.current_nav == "Evaluate":
    render_evaluate_view()
elif st.session_state.current_nav == "Privacy":
    render_privacy_view()
elif st.session_state.current_nav == "Report":
    render_report_view()
