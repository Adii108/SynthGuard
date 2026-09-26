"""
Stitch View 2: Dataset Upload & Profiling Screen.
Mapped from Stitch Screens:
- 'Dataset — Profiling & Schema' (4695212261db4e779fa8ea90b3316ef3)
- 'Dataset — Empty Upload State' (7ff8305c4a874376b2ab5167146693b2)
"""
import streamlit as st
import pandas as pd
from synthguard.ui.components import render_kpi_bar
from synthguard.profiling import DatasetValidator, DatasetProfiler

def render_dataset_view():
    if st.session_state.df_real is None:
        st.markdown("""
        <div class="stitch-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 36px; margin-bottom: 8px;">📂</div>
            <h2 style="font-family: 'Source Serif 4', serif; color: #06271F; margin-bottom: 8px;">No Dataset Ingested Yet</h2>
            <p style="font-size: 14px; color: #717975; max-width: 500px; margin: 0 auto 20px auto;">
                Please upload a CSV dataset or select one of our pre-built benchmark datasets from the Overview page to begin profiling.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Go to Overview to Load Benchmark Dataset", use_container_width=True):
                st.session_state.current_nav = "Overview"
                st.rerun()
        with col2:
            uploaded = st.file_uploader("Upload CSV Directly:", type=["csv"])
            if uploaded:
                val = DatasetValidator()
                res = val.validate_file(uploaded)
                if res.is_valid:
                    st.session_state.df_real = res.dataframe
                    st.session_state.dataset_name = uploaded.name
                    st.session_state.profile = DatasetProfiler().profile(res.dataframe)
                    st.rerun()
        return

    df = st.session_state.df_real
    profile = st.session_state.profile

    st.markdown(f"""
    <div>
        <h1 class="stitch-display">Dataset Profiling & Schema Discovery</h1>
        <p class="stitch-subhead">Inspection for active dataset: <b>{st.session_state.dataset_name}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # 5-Metric KPI Bar
    kpi_data = [
        ("Total Rows", f"{profile.row_count:,}", "Training samples"),
        ("Total Columns", f"{profile.column_count}", "Input features"),
        ("Numerical Features", f"{len(profile.numerical_columns)}", "Continuous / Discrete"),
        ("Categorical Features", f"{len(profile.categorical_columns)}", "Nominal / Ordinal"),
        ("Missing Cells", f"{profile.missing_cell_percentage}%", f"{profile.missing_cell_count} total cells")
    ]
    render_kpi_bar(kpi_data)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Tabular Data Preview", "🔍 Inferred Semantic Schemas", "📈 Marginal Distributions"])

    with tab1:
        st.dataframe(df.head(25), use_container_width=True)
        st.caption(f"Displaying top 25 records out of {len(df):,} total rows.")

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
            st.markdown("#### Numerical Columns Summary")
            if profile.numerical_summary:
                st.dataframe(pd.DataFrame(profile.numerical_summary).T, use_container_width=True)
            else:
                st.info("No numerical continuous columns.")
        with c2:
            st.markdown("#### Categorical Columns Distribution")
            for cat_col in profile.categorical_columns[:3]:
                top_cats = profile.column_profiles[cat_col].stats.get("top_categories", {})
                if top_cats:
                    st.caption(f"**{cat_col}**")
                    st.bar_chart(pd.Series(top_cats))

    st.markdown("---")
    if st.button("Proceed to Natural Language Generation →", type="primary", use_container_width=True):
        st.session_state.current_nav = "Generate"
        st.rerun()
