import synthguard.utils.numpy_compat  # Applies NumPy 2.x compatibility patches
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time

from synthguard.profiling import DatasetValidator, DatasetProfiler
from synthguard.generators import get_generator, GENERATORS
from synthguard.nlp import LLMService, GenerationConfig
from synthguard.reporting import EvaluationPipeline, ReportGenerator

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SynthGuard - Synthetic Data & Privacy Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-med {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

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
    st.session_state.current_nav = "Dashboard"

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ **SynthGuard**")
    st.caption("GenAI Synthetic Tabular Studio")
    st.markdown("---")

    nav_options = ["Dashboard", "1. Dataset Profiling", "2. Generate Data", "3. Evaluate Fidelity & Privacy", "4. AI Final Report"]
    
    # Map friendly names
    nav_map = {
        "Dashboard": "Dashboard",
        "1. Dataset Profiling": "Dataset",
        "2. Generate Data": "Generate",
        "3. Evaluate Fidelity & Privacy": "Evaluate",
        "4. AI Final Report": "Report"
    }
    
    # Determine current index
    default_idx = 0
    for idx, (label, val) in enumerate(nav_map.items()):
        if val == st.session_state.current_nav:
            default_idx = idx
            break

    selected_page_label = st.radio("Navigation Flow:", nav_options, index=default_idx)
    st.session_state.current_nav = nav_map[selected_page_label]

    st.markdown("---")
    st.markdown("### 📁 Dataset Status")
    if st.session_state.df_real is not None:
        st.success(f"**Loaded:** {st.session_state.dataset_name}")
        st.write(f"📊 **Rows:** {len(st.session_state.df_real):,}")
        st.write(f"📐 **Cols:** {len(st.session_state.df_real.columns)}")
    else:
        st.info("No dataset loaded.")

    if st.session_state.df_synth is not None:
        st.markdown("---")
        st.markdown("### 🧪 Synthetic Status")
        st.success(f"Generated: {len(st.session_state.df_synth):,} rows")

    st.markdown("---")
    st.caption("SynthGuard v1.0.0 • Deep Generative Tabular AI")

# ---------------------------------------------------------
# Page 1: DASHBOARD / LANDING PAGE
# ---------------------------------------------------------
if st.session_state.current_nav == "Dashboard":
    st.markdown('<div class="main-title">🛡️ SynthGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate realistic synthetic tabular data while evaluating its utility and privacy risks.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 Synthetic Data</h4>
            <p>Generate high-fidelity tabular data with CTGAN, TVAE, and Gaussian Copula models driven by natural language requests.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        col2.markdown("""
        <div class="metric-card">
            <h4>📈 Data Utility (TSTR)</h4>
            <p>Benchmark downstream ML models using "Train on Synthetic, Test on Real" to guarantee predictive fidelity.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        col3.markdown("""
        <div class="metric-card">
            <h4>🔒 Privacy Analysis</h4>
            <p>Detect exact duplicates, distance to closest record (DCR), and potential training set memorization risks.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start: Select or Upload a Dataset")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Option A: Load Built-in Benchmark Dataset")
        demo_choice = st.selectbox(
            "Choose a sample dataset:",
            ["Employee Attrition (HR Analytics)", "Customer Churn (Telecom)", "Heart Disease (Medical Risk)"]
        )

        if st.button("Load Sample Dataset", type="primary", use_container_width=True):
            if "Employee" in demo_choice:
                df = pd.read_csv("data/employee_attrition.csv")
                name = "Employee Attrition (Demo)"
            elif "Customer" in demo_choice:
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

    with col_b:
        st.markdown("#### Option B: Upload Your Own CSV File")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            validator = DatasetValidator()
            val_res = validator.validate_file(uploaded_file)
            if val_res.is_valid:
                st.session_state.df_real = val_res.dataframe
                st.session_state.dataset_name = uploaded_file.name
                profiler = DatasetProfiler()
                st.session_state.profile = profiler.profile(val_res.dataframe)
                st.success("File uploaded and validated successfully!")
                if st.button("Proceed to Dataset Profiling →", use_container_width=True):
                    st.session_state.current_nav = "Dataset"
                    st.rerun()
            else:
                st.error(f"Validation Error: {', '.join(val_res.errors)}")

    st.markdown("---")
    st.markdown("### 🔄 End-to-End Workflow")
    st.markdown("""
    1. **Upload Dataset** → Validate schema, detect missing values, and profile distributions.
    2. **Natural Language Request** → Describe generation goals (e.g. *\"Generate 1,000 employee records preserving age and salary correlations\"*).
    3. **Synthetic Generation** → Train generative models (CTGAN, TVAE, Gaussian Copula) to synthesize new records.
    4. **Comprehensive Evaluation** → Quantify statistical similarity, correlation retention, TSTR ML utility, and empirical privacy risks.
    5. **Final AI Audit Report** → Export data science reports and synthetic CSV data.
    """)

# ---------------------------------------------------------
# Page 2: DATASET PROFILING
# ---------------------------------------------------------
elif st.session_state.current_nav == "Dataset":
    st.markdown('<div class="main-title">📊 Dataset Ingestion & Profiling</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Inspecting: <b>{st.session_state.dataset_name}</b></div>', unsafe_allow_html=True)

    if st.session_state.df_real is None:
        st.warning("Please upload or select a dataset from the Dashboard first.")
        if st.button("← Go to Dashboard"):
            st.session_state.current_nav = "Dashboard"
            st.rerun()
    else:
        df = st.session_state.df_real
        profile = st.session_state.profile

        # Summary KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Rows", f"{profile.row_count:,}")
        k2.metric("Total Columns", f"{profile.column_count}")
        k3.metric("Numerical Columns", f"{len(profile.numerical_columns)}")
        k4.metric("Categorical Columns", f"{len(profile.categorical_columns)}")
        k5.metric("Missing Cells", f"{profile.missing_cell_percentage}%")

        st.markdown("---")
        
        # Tabs for detailed inspection
        tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "🔍 Column Schemas & Inferred Types", "📈 Statistical Summary"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Showing first 20 rows of {len(df):,} total rows.")

        with tab2:
            schema_data = []
            for col_name, c_prof in profile.column_profiles.items():
                schema_data.append({
                    "Column": col_name,
                    "Inferred Type": c_prof.inferred_type.replace("_", " ").title(),
                    "Native Dtype": c_prof.dtype,
                    "Unique Values": c_prof.unique_count,
                    "Missing %": f"{c_prof.missing_percentage}%",
                    "Sample Values": str(c_prof.sample_values[:3])
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

        with tab3:
            col_num, col_cat = st.columns(2)
            with col_num:
                st.markdown("#### Numerical Columns Statistics")
                if profile.numerical_summary:
                    st.dataframe(pd.DataFrame(profile.numerical_summary).T, use_container_width=True)
                else:
                    st.info("No continuous numerical columns detected.")

            with col_cat:
                st.markdown("#### Categorical Columns Distribution")
                for cat_col in profile.categorical_columns[:4]:
                    top_cats = profile.column_profiles[cat_col].stats.get("top_categories", {})
                    if top_cats:
                        st.caption(f"**{cat_col}** (Top Categories)")
                        st.bar_chart(pd.Series(top_cats))

        st.markdown("---")
        if st.button("Continue to Natural Language Generation →", type="primary", use_container_width=True):
            st.session_state.current_nav = "Generate"
            st.rerun()

# ---------------------------------------------------------
# Page 3: GENERATE SYNTHETIC DATA
# ---------------------------------------------------------
elif st.session_state.current_nav == "Generate":
    st.markdown('<div class="main-title">🧠 Natural Language Generation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Describe your synthetic data goals in natural language and configure generative models.</div>', unsafe_allow_html=True)

    if st.session_state.df_real is None:
        st.warning("Please upload or select a dataset from the Dashboard first.")
    else:
        df = st.session_state.df_real
        available_cols = list(df.columns)

        st.markdown("### 1. Natural Language Intent")
        default_prompt = f"Generate {min(1000, len(df)*2)} synthetic records similar to this dataset while preserving relationships between {', '.join(available_cols[:min(4, len(available_cols))])}."
        
        user_prompt = st.text_area(
            "What kind of synthetic data do you want?",
            value=default_prompt,
            height=100,
            help="Specify desired row count, target objectives, important columns, or privacy priorities in plain English."
        )

        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            if st.button("✨ Interpret & Parse Request with GenAI", type="secondary", use_container_width=True):
                llm = LLMService()
                st.session_state.gen_config = llm.interpret_request(user_prompt, available_cols, len(df))
                st.success("Natural language request successfully parsed!")

        # Show or auto-initialize parsed config
        if st.session_state.gen_config is None:
            llm = LLMService()
            st.session_state.gen_config = llm.interpret_request(user_prompt, available_cols, len(df))

        config: GenerationConfig = st.session_state.gen_config

        st.markdown("---")
        st.markdown("### 2. Interpreted Generation Configuration")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Generation Objective:**\n\n`{config.objective}`")
            st.markdown(f"**Privacy Posture:**\n\n`{config.privacy_priority}`")
        with c2:
            st.markdown(f"**Target Column:**\n\n`{config.target_column}`")
            st.markdown(f"**Key Focus Columns:**\n\n`{', '.join(config.important_columns)}`")
        with c3:
            st.markdown(f"**Number of Records:**\n\n`{config.num_records:,} rows`")
            st.markdown(f"**Recommended Model:**\n\n`{config.selected_model}`")

        st.caption(f"💡 **AI Reasoning:** {config.reasoning}")

        st.markdown("---")
        st.markdown("### 3. Model Execution Settings")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            model_options = list(GENERATORS.keys())
            model_idx = model_options.index(config.selected_model) if config.selected_model in model_options else 0
            chosen_model = st.selectbox("Generative Model Architecture:", model_options, index=model_idx)
        with m_col2:
            row_count = st.number_input("Number of Synthetic Rows:", min_value=20, max_value=20000, value=config.num_records, step=50)
        with m_col3:
            random_seed = st.number_input("Random Seed:", value=42, step=1)

        st.markdown("---")
        
        if st.button("🚀 Train Model & Generate Synthetic Dataset", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing generative model...")
            try:
                time.sleep(0.2)
                progress_bar.progress(20, text=f"Fitting {chosen_model} on {len(df)} training records...")
                
                cat_cols = st.session_state.profile.categorical_columns
                num_cols = st.session_state.profile.numerical_columns

                generator = get_generator(chosen_model, random_state=random_seed)
                generator.fit(df, categorical_columns=cat_cols, numerical_columns=num_cols, epochs=30)
                
                progress_bar.progress(70, text=f"Sampling {row_count} synthetic tabular rows...")
                df_synth = generator.sample(int(row_count))
                
                progress_bar.progress(100, text="Generation complete!")
                st.session_state.df_synth = df_synth
                st.session_state.chosen_model = chosen_model
                
                st.success(f"Successfully generated {len(df_synth):,} synthetic records with {chosen_model}!")
                
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

        if st.session_state.df_synth is not None:
            st.markdown("---")
            st.markdown("### 4. Synthetic Dataset Preview")
            st.dataframe(st.session_state.df_synth.head(20), use_container_width=True)

            csv_buffer = io.StringIO()
            st.session_state.df_synth.to_csv(csv_buffer, index=False)
            
            d_col1, d_col2 = st.columns([1, 1])
            with d_col1:
                st.download_button(
                    label="📥 Download Synthetic CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"synthetic_{st.session_state.dataset_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_col2:
                if st.button("Proceed to Evaluation Suite →", type="primary", use_container_width=True):
                    st.session_state.current_nav = "Evaluate"
                    st.rerun()

# ---------------------------------------------------------
# Page 4: EVALUATE FIDELITY & PRIVACY
# ---------------------------------------------------------
elif st.session_state.current_nav == "Evaluate":
    st.markdown('<div class="main-title">📈 Comprehensive Evaluation Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical Fidelity, Correlation Retention, ML Utility (TSTR), and Empirical Privacy Risk Analysis.</div>', unsafe_allow_html=True)

    if st.session_state.df_real is None or st.session_state.df_synth is None:
        st.warning("Please ensure both real and synthetic datasets are ready. Go to Generate Data first.")
    else:
        df_real = st.session_state.df_real
        df_synth = st.session_state.df_synth
        target_col = st.session_state.gen_config.target_column if st.session_state.gen_config else df_real.columns[-1]
        model_name = getattr(st.session_state, "chosen_model", "CTGAN")

        # Run pipeline if not already run or if triggered
        if st.session_state.evaluation_summary is None or st.button("🔄 Re-run Full Evaluation Suite"):
            with st.spinner("Computing statistical metrics, correlation structures, ML utility benchmarks, and privacy distances..."):
                pipeline = EvaluationPipeline(random_state=42)
                st.session_state.evaluation_summary = pipeline.run(
                    df_real, df_synth, target_column=target_col, selected_model=model_name
                )
            st.success("Evaluation complete!")

        summary = st.session_state.evaluation_summary
        stat = summary.statistical
        corr = summary.correlation
        ml = summary.ml_utility
        priv = summary.privacy

        # High-level score banner
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Statistical Similarity", f"{stat.overall_similarity_score:.1%}")
        c2.metric("Correlation Fidelity", f"{corr.correlation_fidelity_score:.1%}")
        c3.metric("ML Utility Retention", f"{ml.utility_retention_score:.1%}")
        
        # Risk Badge
        risk_class = "badge-low" if "Low" in priv.risk_level else ("badge-med" if "Moderate" in priv.risk_level else "badge-high")
        with c4:
            st.markdown(f"**Privacy Risk Assessment:**\n\n<span class='{risk_class}'>{priv.risk_level}</span>", unsafe_allow_html=True)

        st.markdown("---")

        tab_stat, tab_corr, tab_ml, tab_priv = st.tabs([
            "📊 1. Statistical Similarity",
            "🔗 2. Correlation Preservation",
            "🤖 3. ML Utility (TSTR)",
            "🔒 4. Privacy Risk & Memorization"
        ])

        # Tab 1: Statistical Similarity
        with tab_stat:
            st.markdown("### Marginal Feature Distribution Comparison")
            st.write(f"Evaluates Kolmogorov-Smirnov (KS) distance for numerical features and Total Variation Distance (TVD) for categorical features.")

            sel_col = st.selectbox("Inspect Distribution for Feature:", list(df_real.columns))
            
            p_col1, p_col2 = st.columns([2, 1])
            with p_col1:
                is_num = pd.api.types.is_numeric_dtype(df_real[sel_col]) and df_real[sel_col].nunique() > 5
                if is_num:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df_real[sel_col], name='Real Data', opacity=0.6, marker_color='#3b82f6', histnorm='probability density'))
                    fig.add_trace(go.Histogram(x=df_synth[sel_col], name='Synthetic Data', opacity=0.6, marker_color='#10b981', histnorm='probability density'))
                    fig.update_layout(barmode='overlay', title=f"Real vs Synthetic Density: {sel_col}", xaxis_title=sel_col, yaxis_title="Density")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    df_r_cat = df_real[sel_col].astype(str).value_counts(normalize=True).reset_index()
                    df_r_cat.columns = [sel_col, "Proportion"]
                    df_r_cat["Source"] = "Real"

                    df_s_cat = df_synth[sel_col].astype(str).value_counts(normalize=True).reset_index()
                    df_s_cat.columns = [sel_col, "Proportion"]
                    df_s_cat["Source"] = "Synthetic"

                    df_cat_comb = pd.concat([df_r_cat, df_s_cat])
                    fig = px.bar(df_cat_comb, x=sel_col, y="Proportion", color="Source", barmode="group",
                                 title=f"Category Proportions: {sel_col}", color_discrete_map={"Real": "#3b82f6", "Synthetic": "#10b981"})
                    st.plotly_chart(fig, use_container_width=True)

            with p_col2:
                if sel_col in stat.column_scores:
                    sc = stat.column_scores[sel_col]
                    st.markdown(f"#### Score: **{sc.similarity_score:.1%}**")
                    st.write(f"**Metric:** {sc.metric_name}")
                    st.write(f"**Value:** `{sc.metric_value:.4f}`")
                    st.info(sc.summary_text)

            st.markdown("#### Full Feature Metrics Table")
            stat_table = []
            for col, cs in stat.column_scores.items():
                stat_table.append({
                    "Column": col,
                    "Type": cs.column_type.capitalize(),
                    "Similarity": f"{cs.similarity_score:.1%}",
                    "Metric": cs.metric_name,
                    "Value": cs.metric_value,
                    "Assessment": cs.summary_text
                })
            st.dataframe(pd.DataFrame(stat_table), use_container_width=True)

        # Tab 2: Correlation Preservation
        with tab_corr:
            st.markdown("### Pairwise Correlation Matrix Preservation")
            st.write(corr.summary_text)

            c_fig1, c_fig2 = st.columns(2)
            with c_fig1:
                df_real_corr = pd.DataFrame(corr.real_corr_matrix)
                fig_r = px.imshow(df_real_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Real Data Correlation Matrix")
                st.plotly_chart(fig_r, use_container_width=True)

            with c_fig2:
                df_synth_corr = pd.DataFrame(corr.synth_corr_matrix)
                fig_s = px.imshow(df_synth_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Synthetic Data Correlation Matrix")
                st.plotly_chart(fig_s, use_container_width=True)

            col_p, col_d = st.columns(2)
            with col_p:
                st.markdown("#### ✅ Top Preserved Relationships")
                st.dataframe(pd.DataFrame(corr.preserved_pairs), use_container_width=True)
            with col_d:
                st.markdown("#### ⚠️ Attenuated / Degraded Relationships")
                if corr.degraded_pairs:
                    st.dataframe(pd.DataFrame(corr.degraded_pairs), use_container_width=True)
                else:
                    st.success("No significant correlation degradation observed!")

        # Tab 3: ML Utility (TSTR)
        with tab_ml:
            st.markdown("### Downstream Machine Learning Utility: Train on Synthetic, Test on Real (TSTR)")
            st.info(f"**Concept:** Models are trained strictly on generated data and tested on unseen real data. High predictive performance confirms the synthetic dataset retains genuine decision boundaries.")

            st.write(f"**Target Feature:** `{ml.target_column}` | **Task:** `{ml.task_type.capitalize()}` | **Model:** `{ml.model_name}`")

            ml_comp = []
            if ml.task_type == "classification":
                for k in ["accuracy", "f1_score", "precision", "recall"]:
                    trtr_v = ml.trtr_metrics.get(k, 0.0)
                    tstr_v = ml.tstr_metrics.get(k, 0.0)
                    ret = (tstr_v / trtr_v) if trtr_v > 0 else 0.0
                    ml_comp.append({
                        "Metric": k.replace("_", " ").capitalize(),
                        "Train on Real (TRTR Baseline)": f"{trtr_v:.4f}",
                        "Train on Synthetic (TSTR)": f"{tstr_v:.4f}",
                        "Utility Retention": f"{ret:.1%}"
                    })
            else:
                for k in ["r2_score", "rmse", "mae"]:
                    ml_comp.append({
                        "Metric": k.upper(),
                        "Train on Real (TRTR Baseline)": f"{ml.trtr_metrics.get(k, 0.0):.4f}",
                        "Train on Synthetic (TSTR)": f"{ml.tstr_metrics.get(k, 0.0):.4f}",
                        "Utility Retention": "Baseline vs TSTR"
                    })
            st.dataframe(pd.DataFrame(ml_comp), use_container_width=True)
            st.caption(ml.summary_text)

        # Tab 4: Privacy Risk & Memorization
        with tab_priv:
            st.markdown("### Empirical Privacy & Memorization Risk Analysis")
            st.write("Evaluating proximity of synthetic data points to private real training records.")

            pr1, pr2, pr3, pr4 = st.columns(4)
            pr1.metric("Exact Real Matches", f"{priv.exact_matches_count} ({priv.exact_match_percentage:.1f}%)")
            pr2.metric("Min Distance to Real (DCR)", f"{priv.min_dcr:.4f}")
            pr3.metric("5th Percentile DCR", f"{priv.dcr_5th_percentile:.4f}")
            pr4.metric("Memorization Outliers", f"{priv.potential_memorization_count}")

            st.markdown("---")
            
            col_dcr_chart, col_dcr_text = st.columns([2, 1])
            with col_dcr_chart:
                fig_dcr = px.histogram(
                    priv.dcr_distribution_sample,
                    nbins=25,
                    title="Distance to Closest Record (DCR) Distribution",
                    labels={"value": "Euclidean Distance in Normalized Space", "count": "Frequency"},
                    color_discrete_sequence=["#8b5cf6"]
                )
                st.plotly_chart(fig_dcr, use_container_width=True)

            with col_dcr_text:
                st.markdown("#### Key Findings")
                for f in priv.key_findings:
                    st.write(f"- {f}")
                st.markdown("#### Recommendations")
                for r in priv.recommendations:
                    st.caption(f"• {r}")

        st.markdown("---")
        if st.button("Generate Final AI Audit Report →", type="primary", use_container_width=True):
            st.session_state.current_nav = "Report"
            st.rerun()

# ---------------------------------------------------------
# Page 5: AI FINAL REPORT
# ---------------------------------------------------------
elif st.session_state.current_nav == "Report":
    st.markdown('<div class="main-title">📄 SynthGuard Final Evaluation Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated AI-generated technical & executive audit summary.</div>', unsafe_allow_html=True)

    if st.session_state.evaluation_summary is None:
        st.warning("Evaluation summary not found. Please run the evaluation suite first.")
    else:
        summary = st.session_state.evaluation_summary
        config = st.session_state.gen_config
        report_gen = ReportGenerator()
        
        md_report = report_gen.generate_markdown_report(summary, config, st.session_state.dataset_name)
        html_report = report_gen.generate_html_report(summary, config, st.session_state.dataset_name)

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.download_button(
                label="📥 Download Markdown Report (.md)",
                data=md_report,
                file_name=f"synthguard_report_{st.session_state.dataset_name.replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with r_col2:
            st.download_button(
                label="📥 Download Standalone HTML Report (.html)",
                data=html_report,
                file_name=f"synthguard_report_{st.session_state.dataset_name.replace(' ', '_').lower()}.html",
                mime="text/html",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown(md_report)
