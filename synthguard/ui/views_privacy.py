"""
Stitch View 5: Privacy Risk & Memorization Analysis Screen (Refined).
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from synthguard.ui.components import render_kpi_bar
from synthguard.reporting import EvaluationPipeline

def render_privacy_view():
    if st.session_state.df_real is None or st.session_state.df_synth is None:
        st.warning("Please ensure both real and synthetic datasets are ready. Go to Generate Data first.")
        return

    if st.session_state.evaluation_summary is None:
        with st.spinner("Computing privacy risks & empirical distance metrics..."):
            pipeline = EvaluationPipeline(random_state=42)
            target_col = st.session_state.gen_config.target_column if st.session_state.gen_config else st.session_state.df_real.columns[-1]
            model_name = getattr(st.session_state, "chosen_model", "CTGAN")
            st.session_state.evaluation_summary = pipeline.run(
                st.session_state.df_real, st.session_state.df_synth, target_column=target_col, selected_model=model_name
            )

    priv = st.session_state.evaluation_summary.privacy

    st.markdown("""
    <div style="margin-top: 10px; margin-bottom: 20px;">
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: #E5EDE7; border: 1px solid #C8D6CC; border-radius: 9999px; font-size: 11px; font-weight: 700; color: #0A2E24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: #3A6753;"></span>
            Stage 04 · Empirical Privacy & Memorization Analysis
        </div>
        <h1 class="stitch-display" style="margin-top: 4px;">Empirical Privacy & Memorization Analysis</h1>
        <p class="stitch-subhead">Evaluating geometric proximity, exact duplicate hashing, and potential training data memorization signatures.</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk badge mapping
    risk_cls = "risk-badge-low" if "Low" in priv.risk_level else ("risk-badge-med" if "Moderate" in priv.risk_level else "risk-badge-high")

    st.markdown(f"""
    <div class="stitch-card" style="display: flex; justify-content: space-between; align-items: center; padding: 16px 22px; margin-bottom: 16px;">
        <span style="font-size: 15px; font-weight: 700; color: #06271F;">Overall Assessed Empirical Privacy Risk Tier:</span>
        <span class="{risk_cls}">{priv.risk_level}</span>
    </div>
    """, unsafe_allow_html=True)

    # 4-KPI High-Contrast Bar
    kpi_data = [
        ("Exact Real Matches", f"{priv.exact_matches_count}", f"{priv.exact_match_percentage:.2f}% duplicate rate"),
        ("Minimum DCR", f"{priv.min_dcr:.4f}", "Closest synthetic record"),
        ("5th Percentile DCR", f"{priv.dcr_5th_percentile:.4f}", f"Mean DCR: {priv.mean_dcr:.4f}"),
        ("Memorization Outliers", f"{priv.potential_memorization_count}", f"Mean NNDR: {priv.mean_nndr:.4f}")
    ]
    render_kpi_bar(kpi_data)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_chart, col_text = st.columns([3, 2])

    with col_chart:
        st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 6px;'>Distance to Closest Record (DCR) Distribution</h3>", unsafe_allow_html=True)
        fig_dcr = px.histogram(
            priv.dcr_distribution_sample,
            nbins=30,
            labels={"value": "Normalized Euclidean Distance to Nearest Real Record", "count": "Frequency"},
            color_discrete_sequence=["#0A2E24"]
        )
        fig_dcr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FFFFFF',
            font=dict(family="Inter", size=12, color="#0D1F1A"),
            title="DCR Geometric Separation Histogram"
        )
        st.plotly_chart(fig_dcr, use_container_width=True)
        st.caption("A healthy DCR distribution exhibits geometric separation from training data (no spike at 0.0).")

    with col_text:
        st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 6px;'>Key Privacy Observations</h3>", unsafe_allow_html=True)
        for f in priv.key_findings:
            st.markdown(f"""
            <div style="font-size: 13px; color: #0D1F1A; font-weight: 500; margin-bottom: 8px; padding: 10px 14px; background: #FFFFFF; border: 1px solid #DCE5DE; border-radius: 6px;">
                • {f}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 6px; margin-top: 14px;'>Deployment Recommendations</h3>", unsafe_allow_html=True)
        for r in priv.recommendations:
            st.markdown(f"""
            <div style="font-size: 12px; color: #2E453B; margin-bottom: 6px; line-height: 1.4;">
                ▸ {r}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Generate Final AI Audit Report Dossier →", type="primary", use_container_width=True):
        st.session_state.current_nav = "Report"
        st.rerun()
