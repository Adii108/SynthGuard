"""
SynthGuard: GenAI Platform for Synthetic Tabular Data Generation, Evaluation & Privacy Risk Analysis.
Integrated with Stitch Design System: 'Calm Editorial Intelligence'.
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

# Apply Stitch Design System CSS
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
# Sidebar Navigation (Stitch-aligned)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="font-size: 24px;">🛡️</span>
        <span style="font-family: 'Source Serif 4', serif; font-size: 20px; font-weight: 600; color: #06271F;">SynthGuard</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Empirical Tabular Synthesis Workspace")
    st.markdown("---")

    nav_options = [
        "Overview",
        "1. Dataset Profiling",
        "2. Generate Data",
        "3. Evaluate Utility",
        "4. Privacy Analysis",
        "5. AI Final Report"
    ]
    
    nav_map = {
        "Overview": "Overview",
        "1. Dataset Profiling": "Dataset",
        "2. Generate Data": "Generate",
        "3. Evaluate Utility": "Evaluate",
        "4. Privacy Analysis": "Privacy",
        "5. AI Final Report": "Report"
    }

    default_idx = 0
    for idx, (label, val) in enumerate(nav_map.items()):
        if val == st.session_state.current_nav:
            default_idx = idx
            break

    selected_label = st.radio("Navigation Flow:", nav_options, index=default_idx)
    st.session_state.current_nav = nav_map[selected_label]

    st.markdown("---")
    st.markdown("### 📁 Ingested Dataset")
    if st.session_state.df_real is not None:
        st.markdown(f"**{st.session_state.dataset_name}**")
        st.write(f"📊 **Rows:** {len(st.session_state.df_real):,}")
        st.write(f"📐 **Columns:** {len(st.session_state.df_real.columns)}")
    else:
        st.info("Awaiting CSV dataset ingestion.")

    if st.session_state.df_synth is not None:
        st.markdown("---")
        st.markdown("### 🧪 Generated Artifact")
        st.success(f"{len(st.session_state.df_synth):,} synthetic records")

    st.markdown("---")
    st.caption("SynthGuard v1.0.0 • Stitch UI Integration")

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
