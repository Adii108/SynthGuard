"""
Stitch View 3: Natural Language Planning & Generation Screen (Refined).
"""
import streamlit as st
import pandas as pd
import io
import time
from synthguard.nlp import LLMService, GenerationConfig
from synthguard.generators import get_generator, GENERATORS

def render_generate_view():
    if st.session_state.df_real is None:
        st.warning("Please ingest a dataset from the Overview or Dataset page first.")
        return

    df = st.session_state.df_real
    available_cols = list(df.columns)

    st.markdown("""
    <div style="margin-top: 10px; margin-bottom: 20px;">
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: #E5EDE7; border: 1px solid #C8D6CC; border-radius: 9999px; font-size: 11px; font-weight: 700; color: #0A2E24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: #3A6753;"></span>
            Stage 02 · Natural Language Intent & Generative AI
        </div>
        <h1 class="stitch-display" style="margin-top: 4px;">Natural Language Planning & Generation</h1>
        <p class="stitch-subhead">Translate user intent into structured generative parameters and execute tabular synthesis.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Natural Language Prompt Box
    st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 8px;'>1. Describe Desired Synthetic Data</h3>", unsafe_allow_html=True)
    default_prompt = f"Generate {min(1000, len(df)*2)} synthetic records similar to this dataset while preserving relationships between {', '.join(available_cols[:min(4, len(available_cols))])}."

    user_prompt = st.text_area(
        "What kind of synthetic data do you want?",
        value=default_prompt,
        height=85,
        help="Specify row counts, target objectives, key focus features, and privacy posture in plain English."
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("✨ Parse Intent with GenAI", use_container_width=True):
            llm = LLMService()
            st.session_state.gen_config = llm.interpret_request(user_prompt, available_cols, len(df))
            st.success("Intent successfully extracted and configured!")

    if st.session_state.gen_config is None:
        llm = LLMService()
        st.session_state.gen_config = llm.interpret_request(user_prompt, available_cols, len(df))

    config: GenerationConfig = st.session_state.gen_config

    # 2. Interpreted Configuration Card (High-Contrast Editorial Card)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 8px;'>2. Interpreted Generation Blueprint</h3>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stitch-card" style="border-left: 4px solid #0A2E24; padding: 20px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
            <div>
                <div style="font-size: 11px; color: #4A6357; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Generation Objective</div>
                <div style="font-size: 14px; font-weight: 600; color: #06271F; margin-top: 4px;">{config.objective}</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #4A6357; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Target Feature</div>
                <div style="font-size: 14px; font-weight: 600; color: #06271F; margin-top: 4px;"><code>{config.target_column}</code></div>
            </div>
            <div>
                <div style="font-size: 11px; color: #4A6357; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Record Volume</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: #06271F; margin-top: 4px;">{config.num_records:,} rows</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #4A6357; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Privacy Posture</div>
                <div style="font-size: 13px; font-weight: 700; color: #083D2A; margin-top: 4px;">{config.privacy_priority}</div>
            </div>
        </div>
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #DCE5DE; font-size: 13px; color: #2E453B; line-height: 1.5;">
            <strong>Key Focus Features:</strong> {', '.join([f'<code>{c}</code>' for c in config.important_columns])}
            <br><span style="color: #4A6357; font-size: 12px;">AI Reasoning: {config.reasoning}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Model Controls & Execution
    st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 8px;'>3. Model Architecture & Synthesis</h3>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        model_options = list(GENERATORS.keys())
        default_idx = model_options.index(config.selected_model) if config.selected_model in model_options else 0
        selected_model = st.selectbox("Generative Model Architecture:", model_options, index=default_idx)
    with m_col2:
        num_rows = st.number_input("Synthetic Row Count:", min_value=20, max_value=20000, value=config.num_records, step=50)
    with m_col3:
        seed = st.number_input("Random Seed (Reproducibility):", value=42, step=1)

    if st.button("🚀 Train Model & Generate Synthetic Dataset", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing generative network...")
        try:
            time.sleep(0.15)
            progress_bar.progress(25, text=f"Fitting {selected_model} on {len(df)} records...")

            cat_cols = st.session_state.profile.categorical_columns
            num_cols = st.session_state.profile.numerical_columns

            generator = get_generator(selected_model, random_state=int(seed))
            generator.fit(df, categorical_columns=cat_cols, numerical_columns=num_cols, epochs=30)

            progress_bar.progress(75, text=f"Drawing {num_rows} synthetic samples...")
            df_synth = generator.sample(int(num_rows))

            progress_bar.progress(100, text="Generation completed successfully!")
            st.session_state.df_synth = df_synth
            st.session_state.chosen_model = selected_model
            st.session_state.evaluation_summary = None  # Reset evaluation for new generation
            st.success(f"Generated {len(df_synth):,} synthetic records with {selected_model}!")

        except Exception as e:
            st.error(f"Generation error: {str(e)}")

    # 4. Output Preview & Download
    if st.session_state.df_synth is not None:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #06271F; font-family: \"Source Serif 4\", serif; font-size: 19px; font-weight: 600; margin-bottom: 8px;'>4. Synthetic Data Artifact Preview</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.df_synth.head(20), use_container_width=True)

        csv_buf = io.StringIO()
        st.session_state.df_synth.to_csv(csv_buf, index=False)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="📥 Download Synthetic CSV",
                data=csv_buf.getvalue(),
                file_name=f"synthetic_{st.session_state.dataset_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            if st.button("Proceed to Comprehensive Evaluation →", type="primary", use_container_width=True):
                st.session_state.current_nav = "Evaluate"
                st.rerun()
