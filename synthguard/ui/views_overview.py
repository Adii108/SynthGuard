"""
Stitch View 1: Overview & Welcome Screen (Refined).
"""
import streamlit as st
import pandas as pd
from synthguard.profiling import DatasetValidator, DatasetProfiler

def render_overview_view():
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: #E5EDE7; border: 1px solid #C8D6CC; border-radius: 9999px; font-size: 11px; font-weight: 700; color: #0A2E24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: #15803D;"></span>
            Empirical Synthesis Engine
        </div>
        <h1 class="stitch-display">Turn real data into useful, privacy-aware synthetic data.</h1>
        <p class="stitch-subhead">
            SynthGuard enables data scientists and research teams to generate realistic tabular datasets from natural language requirements,
            while measuring marginal distribution fidelity, downstream machine learning utility (TSTR), and empirical memorization risks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Editorial Assurance Callout
    st.markdown("""
    <div class="stitch-card-inset" style="display: flex; gap: 14px; align-items: flex-start; margin-bottom: 24px;">
        <span style="font-size: 22px; line-height: 1;">🛡️</span>
        <div style="font-size: 13px; color: #2E453B; line-height: 1.5;">
            <strong style="color: #06271F; font-size: 14px;">No AI hype or synthetic illusions.</strong> Transparent statistical validation (Kolmogorov-Smirnov, 1-Wasserstein, Cramer's V) and empirical privacy guarantees (Distance-to-Closest-Record & NNDR) for high-stakes tabular datasets.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 20px; font-weight: 600; margin-bottom: 14px;'>🚀 Quick-Start Workspace</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="stitch-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin-top: 0; color: #06271F; font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600;">Option A: Pre-Loaded Benchmark Datasets</h4>
                <p style="font-size: 13px; color: #4A6357; margin-bottom: 12px;">Select a realistic domain dataset to immediately test the end-to-end generation and evaluation pipeline.</p>
            </div>
        """, unsafe_allow_html=True)
        
        sample_choice = st.selectbox(
            "Select Benchmark Dataset:",
            [
                "Employee Attrition (HR Analytics, 600 rows)",
                "Customer Churn (Telecom, 500 rows)",
                "Heart Disease (Medical Risk, 400 rows)"
            ]
        )

        if st.button("Load Benchmark Dataset →", use_container_width=True):
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
        <div class="stitch-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin-top: 0; color: #06271F; font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600;">Option B: Upload Your Own CSV</h4>
                <p style="font-size: 13px; color: #4A6357; margin-bottom: 12px;">Upload a raw comma-separated tabular file to automatically inspect columns and data quality.</p>
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
    st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 20px; font-weight: 600; margin-bottom: 14px;'>🔄 Complete 5-Stage Scientific Workflow</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
        <div class="stitch-card" style="padding: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #0A2E24; margin-bottom: 4px;">STAGE 01</div>
            <strong style="color: #06271F; font-size: 14px;">Dataset Ingestion</strong>
            <p style="font-size: 12px; color: #4A6357; margin-top: 6px; line-height: 1.4;">Schema validation, missing value detection, and semantic type classification.</p>
        </div>
        <div class="stitch-card" style="padding: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #0A2E24; margin-bottom: 4px;">STAGE 02</div>
            <strong style="color: #06271F; font-size: 14px;">NL Planning</strong>
            <p style="font-size: 12px; color: #4A6357; margin-top: 6px; line-height: 1.4;">Translate natural language goals into structured generative parameters.</p>
        </div>
        <div class="stitch-card" style="padding: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #0A2E24; margin-bottom: 4px;">STAGE 03</div>
            <strong style="color: #06271F; font-size: 14px;">Generative AI</strong>
            <p style="font-size: 12px; color: #4A6357; margin-top: 6px; line-height: 1.4;">Train CTGAN (VGM), TVAE, or Gaussian Copula generative architectures.</p>
        </div>
        <div class="stitch-card" style="padding: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #0A2E24; margin-bottom: 4px;">STAGE 04</div>
            <strong style="color: #06271F; font-size: 14px;">Dual Evaluation</strong>
            <p style="font-size: 12px; color: #4A6357; margin-top: 6px; line-height: 1.4;">Quantify KS similarity, correlation preservation, and TSTR ML utility.</p>
        </div>
        <div class="stitch-card" style="padding: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #0A2E24; margin-bottom: 4px;">STAGE 05</div>
            <strong style="color: #06271F; font-size: 14px;">Privacy Audit</strong>
            <p style="font-size: 12px; color: #4A6357; margin-top: 6px; line-height: 1.4;">Evaluate exact matches, DCR distances, and export AI audit dossier.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
