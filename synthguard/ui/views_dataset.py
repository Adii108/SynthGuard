"""
Stitch View 2: Dataset Upload & Profiling Screen (Refined & Spaced).
"""
import streamlit as st
import pandas as pd
from synthguard.ui.components import render_kpi_bar
from synthguard.profiling import DatasetValidator, DatasetProfiler

def render_dataset_view():
    if st.session_state.df_real is None:
        st.markdown("""
        <div style="margin-top: 10px; margin-bottom: 24px;">
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: #E5EDE7; border: 1px solid #C8D6CC; border-radius: 9999px; font-size: 11px; font-weight: 700; color: #0A2E24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #3A6753;"></span>
                Stage 01 · Ingestion & Profiling
            </div>
            <h1 class="stitch-display" style="margin-top: 4px;">Let's understand your data first.</h1>
            <p class="stitch-subhead">
                Upload a CSV dataset and SynthGuard will automatically profile its column types, statistical distributions, and data quality before generating synthetic records.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="stitch-card" style="padding: 24px;">
                <h4 style="margin-top: 0; color: #06271F; font-family: 'Source Serif 4', serif; font-size: 18px; font-weight: 600;">Upload Tabular Dataset</h4>
                <p style="font-size: 13px; color: #4A6357; margin-bottom: 16px;">Supports standard comma-separated CSV files up to 200 MB.</p>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader("Drop your CSV file here or browse:", type=["csv"])
            if uploaded:
                val = DatasetValidator()
                res = val.validate_file(uploaded)
                if res.is_valid:
                    st.session_state.df_real = res.dataframe
                    st.session_state.dataset_name = uploaded.name
                    st.session_state.profile = DatasetProfiler().profile(res.dataframe)
                    st.success("Dataset ingested and profiled successfully!")
                    st.rerun()
                else:
                    st.error(f"Validation Error: {', '.join(res.errors)}")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="stitch-card" style="padding: 24px; min-height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h4 style="margin-top: 0; color: #06271F; font-family: 'Source Serif 4', serif; font-size: 18px; font-weight: 600;">Or Load a Sample Dataset</h4>
                    <p style="font-size: 13px; color: #4A6357; margin-bottom: 14px;">Test immediately with a realistic multi-feature domain dataset.</p>
                </div>
            """, unsafe_allow_html=True)

            choice = st.selectbox("Select Benchmark Dataset:", [
                "Employee Attrition (HR Analytics, 600 rows)",
                "Customer Churn (Telecom, 500 rows)",
                "Heart Disease (Medical Risk, 400 rows)"
            ])

            if st.button("Load & Profile Benchmark Data →", use_container_width=True):
                if "Employee" in choice:
                    df = pd.read_csv("data/employee_attrition.csv")
                    name = "Employee Attrition (Demo)"
                elif "Customer" in choice:
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
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
        return

    df = st.session_state.df_real
    profile = st.session_state.profile

    st.markdown(f"""
    <div style="margin-top: 10px; margin-bottom: 20px;">
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: #E5EDE7; border: 1px solid #C8D6CC; border-radius: 9999px; font-size: 11px; font-weight: 700; color: #0A2E24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: #15803D;"></span>
            Dataset Profiled
        </div>
        <h1 class="stitch-display" style="margin-top: 4px;">Dataset Profiling & Schema Discovery</h1>
        <p class="stitch-subhead">Inspection results for active dataset: <b>{st.session_state.dataset_name}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # 5-Metric High-Contrast KPI Bar
    kpi_data = [
        ("Total Rows", f"{profile.row_count:,}", "Training samples"),
        ("Total Columns", f"{profile.column_count}", "Input features"),
        ("Numerical Features", f"{len(profile.numerical_columns)}", "Continuous / Discrete"),
        ("Categorical Features", f"{len(profile.categorical_columns)}", "Nominal / Ordinal"),
        ("Missing Cells", f"{profile.missing_cell_percentage}%", f"{profile.missing_cell_count} null cells")
    ]
    render_kpi_bar(kpi_data)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Tabular Data Preview", "🔍 Inferred Semantic Schemas", "📈 Marginal Distributions"])

    with tab1:
        st.dataframe(df.head(25), use_container_width=True)
        st.caption(f"Displaying top 25 rows out of {len(df):,} total records.")

    with tab2:
        schema_rows = []
        for col, cp in profile.column_profiles.items():
            schema_rows.append({
                "Feature Name": col,
                "Inferred Semantic Type": cp.inferred_type.replace("_", " ").title(),
                "Storage Dtype": cp.dtype,
                "Distinct Values": cp.unique_count,
                "Missing %": f"{cp.missing_percentage}%",
                "Sample Values": str(cp.sample_values[:3])
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<h4 style='color: #06271F; font-size: 16px; font-weight: 600;'>Numerical Continuous Summaries</h4>", unsafe_allow_html=True)
            if profile.numerical_summary:
                st.dataframe(pd.DataFrame(profile.numerical_summary).T, use_container_width=True)
            else:
                st.info("No numerical continuous columns detected.")
        with c2:
            st.markdown("<h4 style='color: #06271F; font-size: 16px; font-weight: 600;'>Categorical Distribution Previews</h4>", unsafe_allow_html=True)
            for cat_col in profile.categorical_columns[:3]:
                top_cats = profile.column_profiles[cat_col].stats.get("top_categories", {})
                if top_cats:
                    st.caption(f"**{cat_col}** (Top Frequencies)")
                    st.bar_chart(pd.Series(top_cats))

    st.markdown("---")
    if st.button("Proceed to Natural Language Generation →", type="primary", use_container_width=True):
        st.session_state.current_nav = "Generate"
        st.rerun()
