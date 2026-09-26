"""
Stitch View 6: Comprehensive AI Evaluation Dossier & Report Export.
Mapped from Stitch Screen: 'Report — Comprehensive Evaluation Dossier' (c6efe2fa4b204a4caef8bfbbb47bc47b).
"""
import streamlit as st
import io
from synthguard.reporting import ReportGenerator

def render_report_view():
    if st.session_state.evaluation_summary is None:
        st.warning("No evaluation results available. Please run the evaluation suite first.")
        return

    summary = st.session_state.evaluation_summary
    config = st.session_state.gen_config
    report_gen = ReportGenerator()

    md_report = report_gen.generate_markdown_report(summary, config, st.session_state.dataset_name)
    html_report = report_gen.generate_html_report(summary, config, st.session_state.dataset_name)

    st.markdown("""
    <div>
        <h1 class="stitch-display">Comprehensive AI Evaluation Dossier</h1>
        <p class="stitch-subhead">Grounded technical and executive audit report synthesized directly from calculated empirical metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Action bar for downloads
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="📥 Download Markdown Dossier (.md)",
            data=md_report,
            file_name=f"synthguard_audit_{st.session_state.dataset_name.replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with c2:
        st.download_button(
            label="📥 Download Standalone HTML Dossier (.html)",
            data=html_report,
            file_name=f"synthguard_audit_{st.session_state.dataset_name.replace(' ', '_').lower()}.html",
            mime="text/html",
            use_container_width=True
        )
    with c3:
        if st.session_state.df_synth is not None:
            csv_buf = io.StringIO()
            st.session_state.df_synth.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Synthetic CSV (.csv)",
                data=csv_buf.getvalue(),
                file_name=f"synthetic_{st.session_state.dataset_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Dossier Document Card
    st.markdown("""
    <div class="stitch-card" style="padding: 28px; background: #FFFFFF;">
    """, unsafe_allow_html=True)

    st.markdown(md_report)

    st.markdown("</div>", unsafe_allow_html=True)
