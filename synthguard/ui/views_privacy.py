"""
Stitch View 5: Privacy Risk & Memorization Analysis Screen.
Mapped from Stitch Screen: 'Privacy — Disclosure Risk Analysis' (207b942a9ab845849e78e7d4ade74bf8).
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
    <div>
        <h1 class="stitch-display">Empirical Privacy & Memorization Analysis</h1>
        <p class="stitch-subhead">Evaluating geometric proximity, exact duplicate hashing, and potential training data memorization signatures.</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk badge mapping
    risk_cls = "risk-badge-low" if "Low" in priv.risk_level else ("risk-badge-med" if "Moderate" in priv.risk_level else "risk-badge-high")

    st.markdown(f"""
    <div class="stitch-card" style="display: flex; justify-content: space-between; align-items: center; padding: 14px 20px;">
        <span style="font-size: 14px; font-weight: 600; color: #06271F;">Overall Assessed Privacy Risk Tier:</span>
        <span class="{risk_cls}">{priv.risk_level}</span>
    </div>
    """, unsafe_allow_html=True)

    # 4-KPI Bar
    kpi_data = [
        ("Exact Real Matches", f"{priv.exact_matches_count}", f"{priv.exact_match_percentage:.2f}% duplicate rate"),
        ("Minimum DCR", f"{priv.min_dcr:.4f}", "Closest synthetic record"),
        ("5th Percentile DCR", f"{priv.dcr_5th_percentile:.4f}", f"Mean DCR: {priv.mean_dcr:.4f}"),
        ("Memorization Outliers", f"{priv.potential_memorization_count}", f"Mean NNDR: {priv.mean_nndr:.4f}")
    ]
    render_kpi_bar(kpi_data)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_chart, col_text = st.columns([3, 2])

    with col_chart:
        st.markdown("### Distance to Closest Record (DCR) Distribution")
        fig_dcr = px.histogram(
            priv.dcr_distribution_sample,
            nbins=30,
            labels={"value": "Normalized Euclidean Distance to Nearest Real Record", "count": "Frequency"},
            color_discrete_sequence=["#3A6753"]
        )
        fig_dcr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FFFFFF',
            font=dict(family="Inter", size=12),
            title="DCR Separation Histogram"
        )
        st.plotly_chart(fig_dcr, use_container_width=True)
        st.caption("A healthy DCR distribution exhibits geometric separation from training data (no spike at 0.0).")

    with col_text:
        st.markdown("### Key Empirical Privacy Findings")
        for f in priv.key_findings:
            st.markdown(f"""
            <div style="font-size: 13px; color: #181D1A; margin-bottom: 8px; padding: 8px 12px; background: #FFFFFF; border: 1px solid #DFE4DF; border-radius: 6px;">
                • {f}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Deployment Recommendations")
        for r in priv.recommendations:
            st.markdown(f"""
            <div style="font-size: 12px; color: #414845; margin-bottom: 6px;">
                ▸ {r}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Generate Final AI Audit Report Dossier →", type="primary", use_container_width=True):
        st.session_state.current_nav = "Report"
        st.rerun()
