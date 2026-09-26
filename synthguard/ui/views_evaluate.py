"""
Stitch View 4: Statistical Similarity, Correlation Preservation & ML Utility Evaluation.
Mapped from Stitch Screen: 'Evaluate — Statistical & ML Utility' (f54f47ec8a294d50885682d064aaa109).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from synthguard.ui.components import render_kpi_bar
from synthguard.reporting import EvaluationPipeline

def render_evaluate_view():
    if st.session_state.df_real is None or st.session_state.df_synth is None:
        st.warning("Please ensure both real and synthetic datasets are ready. Go to Generate Data first.")
        return

    df_real = st.session_state.df_real
    df_synth = st.session_state.df_synth
    target_col = st.session_state.gen_config.target_column if st.session_state.gen_config else df_real.columns[-1]
    model_name = getattr(st.session_state, "chosen_model", "CTGAN")

    # Run pipeline if not already executed
    if st.session_state.evaluation_summary is None:
        with st.spinner("Computing statistical metrics, correlation structures, and TSTR benchmarks..."):
            pipeline = EvaluationPipeline(random_state=42)
            st.session_state.evaluation_summary = pipeline.run(
                df_real, df_synth, target_column=target_col, selected_model=model_name
            )

    summary = st.session_state.evaluation_summary
    stat = summary.statistical
    corr = summary.correlation
    ml = summary.ml_utility

    st.markdown("""
    <div>
        <h1 class="stitch-display">Statistical Fidelity & ML Utility Evaluation</h1>
        <p class="stitch-subhead">Empirical comparison between original dataset and generated synthetic data across marginal, joint, and predictive axes.</p>
    </div>
    """, unsafe_allow_html=True)

    # 3-Metric KPI Scoreboard
    kpi_data = [
        ("Statistical Similarity", f"{stat.overall_similarity_score:.1%}", "Marginal KS / TVD overlap"),
        ("Correlation Retention", f"{corr.correlation_fidelity_score:.1%}", f"MAE: {corr.mean_absolute_error:.4f}"),
        ("ML Utility Retention", f"{ml.utility_retention_score:.1%}", f"TSTR on {ml.model_name}")
    ]
    render_kpi_bar(kpi_data)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 1. Marginal Feature Distributions",
        "🔗 2. Correlation Matrix Preservation",
        "🤖 3. Downstream ML Utility (TSTR)"
    ])

    with tab1:
        st.markdown("### Feature Distribution Overlays")
        st.write("Two-Sample Kolmogorov-Smirnov (KS) Test for continuous features and Total Variation Distance (TVD) for categorical features.")

        sel_col = st.selectbox("Select Feature to Inspect:", list(df_real.columns))
        col_c1, col_c2 = st.columns([2, 1])

        with col_c1:
            is_num = pd.api.types.is_numeric_dtype(df_real[sel_col]) and df_real[sel_col].nunique() > 5
            if is_num:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df_real[sel_col], name='Real Data', opacity=0.6, marker_color='#06271F', histnorm='probability density'
                ))
                fig.add_trace(go.Histogram(
                    x=df_synth[sel_col], name='Synthetic Data', opacity=0.6, marker_color='#3A6753', histnorm='probability density'
                ))
                fig.update_layout(
                    barmode='overlay',
                    title=f"Density Overlay: {sel_col}",
                    xaxis_title=sel_col,
                    yaxis_title="Probability Density",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#FFFFFF',
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                df_r_cat = df_real[sel_col].astype(str).value_counts(normalize=True).reset_index()
                df_r_cat.columns = [sel_col, "Proportion"]
                df_r_cat["Source"] = "Real"

                df_s_cat = df_synth[sel_col].astype(str).value_counts(normalize=True).reset_index()
                df_s_cat.columns = [sel_col, "Proportion"]
                df_s_cat["Source"] = "Synthetic"

                df_cat_comb = pd.concat([df_r_cat, df_s_cat])
                fig = px.bar(
                    df_cat_comb, x=sel_col, y="Proportion", color="Source", barmode="group",
                    title=f"Category Distribution: {sel_col}",
                    color_discrete_map={"Real": "#06271F", "Synthetic": "#3A6753"}
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FFFFFF', font=dict(family="Inter", size=12))
                st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            if sel_col in stat.column_scores:
                sc = stat.column_scores[sel_col]
                st.markdown(f"""
                <div class="stitch-card" style="margin-top: 20px;">
                    <div style="font-size: 11px; color: #717975; text-transform: uppercase; font-weight: 600;">Similarity Score</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: #06271F; margin: 4px 0;">{sc.similarity_score:.1%}</div>
                    <div style="font-size: 12px; color: #414845; margin-bottom: 8px;"><b>Metric:</b> {sc.metric_name} (`{sc.metric_value:.4f}`)</div>
                    <div style="font-size: 12px; color: #717975; line-height: 1.4;">{sc.summary_text}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Complete Feature Similarity Table")
        stat_rows = []
        for col, cs in stat.column_scores.items():
            stat_rows.append({
                "Feature": col,
                "Type": cs.column_type.capitalize(),
                "Similarity Score": f"{cs.similarity_score:.1%}",
                "Metric Name": cs.metric_name,
                "Metric Value": cs.metric_value,
                "Qualitative Assessment": cs.summary_text
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

    with tab2:
        st.markdown("### Pairwise Correlation & Association Preservation")
        st.write(corr.summary_text)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            df_real_corr = pd.DataFrame(corr.real_corr_matrix)
            fig_r = px.imshow(
                df_real_corr, text_auto=".2f", color_continuous_scale="Tealgrn", zmin=-1, zmax=1, title="Real Data Correlation Matrix"
            )
            fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", size=11))
            st.plotly_chart(fig_r, use_container_width=True)

        with col_h2:
            df_synth_corr = pd.DataFrame(corr.synth_corr_matrix)
            fig_s = px.imshow(
                df_synth_corr, text_auto=".2f", color_continuous_scale="Tealgrn", zmin=-1, zmax=1, title="Synthetic Data Correlation Matrix"
            )
            fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", size=11))
            st.plotly_chart(fig_s, use_container_width=True)

        col_pr, col_deg = st.columns(2)
        with col_pr:
            st.markdown("#### ✅ Preserved Feature Relationships")
            st.dataframe(pd.DataFrame(corr.preserved_pairs), use_container_width=True)
        with col_deg:
            st.markdown("#### ⚠️ Attenuated / Drifted Feature Relationships")
            if corr.degraded_pairs:
                st.dataframe(pd.DataFrame(corr.degraded_pairs), use_container_width=True)
            else:
                st.success("No significant correlation degradation observed!")

    with tab3:
        st.markdown("### Downstream Machine Learning Utility: Train on Synthetic, Test on Real (TSTR)")
        st.markdown("""
        <div class="stitch-card-inset" style="font-size: 13px; color: #414845;">
            <strong>Methodology:</strong> We train predictive models exclusively on the <i>generated synthetic dataset</i> and evaluate their generalization on unseen <i>real test data</i>. 
            Comparing these scores against the empirical <i>Train on Real, Test on Real (TRTR)</i> baseline measures how well the synthetic data preserves true decision boundaries.
        </div>
        """, unsafe_allow_html=True)

        st.write(f"**Target Column:** ` {ml.target_column} ` | **Task Type:** `{ml.task_type.capitalize()}` | **Algorithm:** `{ml.model_name}`")

        ml_rows = []
        if ml.task_type == "classification":
            for m in ["accuracy", "f1_score", "precision", "recall"]:
                trtr_v = ml.trtr_metrics.get(m, 0.0)
                tstr_v = ml.tstr_metrics.get(m, 0.0)
                ret = (tstr_v / trtr_v) if trtr_v > 0 else 0.0
                ml_rows.append({
                    "Downstream Metric": m.replace("_", " ").title(),
                    "Train on Real (TRTR Baseline)": f"{trtr_v:.4f}",
                    "Train on Synthetic (TSTR)": f"{tstr_v:.4f}",
                    "Predictive Utility Retention": f"{ret:.1%}"
                })
        else:
            for m in ["r2_score", "rmse", "mae"]:
                ml_rows.append({
                    "Downstream Metric": m.upper(),
                    "Train on Real (TRTR Baseline)": f"{ml.trtr_metrics.get(m, 0.0):.4f}",
                    "Train on Synthetic (TSTR)": f"{ml.tstr_metrics.get(m, 0.0):.4f}",
                    "Predictive Utility Retention": "Baseline vs TSTR"
                })
        st.dataframe(pd.DataFrame(ml_rows), use_container_width=True)

    st.markdown("---")
    if st.button("Proceed to Privacy Risk Analysis →", type="primary", use_container_width=True):
        st.session_state.current_nav = "Privacy"
        st.rerun()
