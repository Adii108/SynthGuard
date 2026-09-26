"""
Stitch View 1: Overview & Welcome Screen.
Mapped from Stitch Screen: 'Overview — Welcome to SynthGuard' (070a785b1d9f46e6a40c0b0c66875076).
"""
import streamlit as st
import pandas as pd
from synthguard.profiling import DatasetValidator, DatasetProfiler

def render_overview_view():
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: #E5E9E4; border-radius: 9999px; font-size: 11px; font-weight: 600; color: #3A6753; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: #3A6753;"></span>
            Empirical Synthesis Engine
        </div>
        <h1 class="stitch-display">Turn real data into useful, privacy-aware synthetic data.</h1>
        <p class="stitch-subhead">
            SynthGuard allows data teams and researchers to synthesize high-fidelity tabular data from natural language instructions, 
            while systematically evaluating statistical fidelity, downstream ML utility, and empirical privacy risks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Assurance Badge
    st.markdown("""
    <div class="stitch-card-inset" style="display: flex; gap: 12px; align-items: flex-start;">
        <span style="font-size: 20px;">🛡️</span>
        <div style="font-size: 13px; color: #414845; line-height: 1.5;">
            <strong style="color: #06271F;">No AI hype.</strong> Transparent statistical verification (Kolmogorov-Smirnov, 1-Wasserstein, Cramer's V) and empirical memorization guarantees (Distance-to-Closest-Record & NNDR) for high-stakes tabular datasets.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Quick-Start Options")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="stitch-card" style="min-height: 240px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin-top: 0; color: #06271F;">Option A: Built-in Benchmark Datasets</h4>
                <p style="font-size: 13px; color: #717975;">Instantly load pre-configured realistic tabular benchmarks for instant demonstration.</p>
            </div>
        """, unsafe_allow_html=True)
        
        sample_choice = st.selectbox(
            "Select Benchmark Dataset:",
            ["Employee Attrition (HR Analytics, 600 rows)", "Customer Churn (Telecom, 500 rows)", "Heart Disease (Medical Risk, 400 rows)"]
        )

        if st.button("Load Selected Dataset →", use_container_width=True):
            if "Employee" in sample_choice:
                df = pd.read_csv("data/employee_attrition.csv")
                name = "Employee Attrition (Demo)"
            elif "Customer" in sample_choice:
                df = pd.read_csv("data/customer_churn.csv")
                name = "Customer Churn (Demo)"
            else:
                df = pd.read_csv("data/heart_disease.csv")
                name = "Heart Disease (Demo)"

            validator = DatasetValidator()
            val_res = validator.validate_dataframe(df)
            if val_res.is_valid:
                st.session_state.df_real = df
                st.session_state.dataset_name = name
                profiler = DatasetProfiler()
                st.session_state.profile = profiler.profile(df)
                st.session_state.current_nav = "Dataset"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stitch-card" style="min-height: 240px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin-top: 0; color: #06271F;">Option B: Upload Custom CSV</h4>
                <p style="font-size: 13px; color: #717975;">Upload your proprietary tabular CSV dataset to begin profiling and synthesis.</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            validator = DatasetValidator()
            val_res = validator.validate_file(uploaded_file)
            if val_res.is_valid:
                st.session_state.df_real = val_res.dataframe
                st.session_state.dataset_name = uploaded_file.name
                profiler = DatasetProfiler()
                st.session_state.profile = profiler.profile(val_res.dataframe)
                st.success("File validated successfully!")
                if st.button("Inspect Dataset Schema & Profiling →", use_container_width=True):
                    st.session_state.current_nav = "Dataset"
                    st.rerun()
            else:
                st.error(f"Validation Error: {', '.join(val_res.errors)}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 5-Stage Scientific Workflow")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
        <div class="stitch-card" style="padding: 14px;">
            <strong style="color: #06271F;">1. Dataset Ingestion</strong>
            <p style="font-size: 12px; color: #717975; margin-top: 4px;">CSV schema validation, missing cell detection, and semantic typing.</p>
        </div>
        <div class="stitch-card" style="padding: 14px;">
            <strong style="color: #06271F;">2. NL Planning</strong>
            <p style="font-size: 12px; color: #717975; margin-top: 4px;">Convert natural language requirements into generative parameters.</p>
        </div>
        <div class="stitch-card" style="padding: 14px;">
            <strong style="color: #06271F;">3. Generative AI</strong>
            <p style="font-size: 12px; color: #717975; margin-top: 4px;">Train CTGAN, TVAE, or Gaussian Copula generative architectures.</p>
        </div>
        <div class="stitch-card" style="padding: 14px;">
            <strong style="color: #06271F;">4. Dual Evaluation</strong>
            <p style="font-size: 12px; color: #717975; margin-top: 4px;">Measure statistical similarity, correlation retention, and TSTR ML utility.</p>
        </div>
        <div class="stitch-card" style="padding: 14px;">
            <strong style="color: #06271F;">5. Privacy Audit</strong>
            <p style="font-size: 12px; color: #717975; margin-top: 4px;">Quantify exact duplicates, DCR distances, and export AI audit dossier.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
