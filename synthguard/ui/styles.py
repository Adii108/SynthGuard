"""
Stitch Design System Tokens & CSS for SynthGuard.
Theme: 'Calm Editorial Intelligence' - Refined High-Contrast Theme
Primary: Deep Forest Green (#0A2E24 / #06271F)
Secondary: Sage Slate (#2D5A47 / #3A6753)
Background: Warm Off-White (#F8FAF8 / #F5F8F5)
Surface: Pure White (#FFFFFF)
Text Primary: Deep Forest Charcoal (#0D1F1A / #0F172A)
Text Secondary: Slate Green (#384D44 / #475569)
"""

STITCH_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
    /* Global Canvas & Base Background */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #F8FAF8 !important;
        color: #0D1F1A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(248, 250, 248, 0.9) !important;
    }

    /* Main Container Padding Adjustment to Prevent Clipping */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar Background & Contrast Fixes */
    [data-testid="stSidebar"] {
        background-color: #EEF3EF !important;
        border-right: 1px solid #D4DFD7 !important;
    }

    [data-testid="stSidebar"] * {
        color: #0D1F1A !important;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #2E453B !important;
        font-size: 13px !important;
    }

    [data-testid="stSidebar"] .stCaption p, [data-testid="stSidebar"] small {
        color: #4A6357 !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }

    /* Radio Button Navigation in Sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #3A5246 !important;
        margin-bottom: 8px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        background: #FFFFFF !important;
        border: 1px solid #D4DFD7 !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #E5EDE7 !important;
        border-color: #B5C8BA !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #0A2E24 !important;
        border-color: #0A2E24 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #0D1F1A !important;
    }

    /* Headings in Source Serif 4 */
    h1, h2, h3, .stitch-title {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #06271F !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }

    .stitch-display {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 2.1rem !important;
        line-height: 1.3 !important;
        font-weight: 500 !important;
        color: #06271F !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    .stitch-subhead {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.02rem !important;
        color: #384D44 !important;
        line-height: 1.6 !important;
        margin-bottom: 1.5rem !important;
    }

    /* Tabular Figures */
    .tnum, [data-testid="stMetricValue"], code {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Top Sticky Header */
    .stitch-header-container {
        background: #FFFFFF !important;
        border: 1px solid #DCE5DE !important;
        padding: 14px 18px !important;
        border-radius: 10px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 1px 4px rgba(10, 46, 36, 0.04) !important;
    }

    .stitch-badge-ready {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 4px 12px !important;
        border-radius: 9999px !important;
        background-color: #D1EAD9 !important;
        color: #083D2A !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    .stitch-badge-ready::before {
        content: '';
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #15803D;
    }

    .stitch-badge-awaiting {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 4px 12px !important;
        border-radius: 9999px !important;
        background-color: #E2EBE5 !important;
        color: #4A6357 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    /* Stepper Bar */
    .stitch-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding-top: 10px;
        border-top: 1px solid #EDF2EE;
        margin-top: 10px;
        overflow-x: auto;
    }

    .stitch-step-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        color: #4A6357;
        text-decoration: none;
        white-space: nowrap;
    }

    .stitch-step-item.active {
        color: #06271F;
        font-weight: 700;
    }

    .stitch-step-num {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #E2EBE5;
        color: #2E453B;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .stitch-step-item.active .stitch-step-num {
        background: #0A2E24;
        color: #FFFFFF;
    }

    .stitch-step-arrow {
        color: #A9BDB0;
        font-size: 12px;
        font-weight: 600;
    }

    /* Surface Cards */
    .stitch-card {
        background: #FFFFFF !important;
        border: 1px solid #DCE5DE !important;
        border-radius: 10px !important;
        padding: 22px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 1px 3px rgba(10, 46, 36, 0.03) !important;
    }

    .stitch-card-inset {
        background: #F1F6F2 !important;
        border: 1px solid #D4DFD7 !important;
        border-radius: 8px !important;
        padding: 16px 18px !important;
        margin-bottom: 14px !important;
    }

    /* Streamlit Input & Widget Contrast */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label, .stFileUploader label {
        color: #0D1F1A !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #0D1F1A !important;
        border: 1px solid #CBD7CF !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0A2E24 !important;
        box-shadow: 0 0 0 2px rgba(10, 46, 36, 0.12) !important;
    }

    /* File Uploader Dropzone Styling */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #B5C8BA !important;
        border-radius: 10px !important;
        padding: 24px !important;
        transition: border-color 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0A2E24 !important;
        background-color: #F8FAF8 !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #2E453B !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background-color: #FFFFFF !important;
        color: #0A2E24 !important;
        border: 1px solid #B5C8BA !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 6px 16px !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #EEF3EF !important;
        border-color: #0A2E24 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0A2E24 !important;
        color: #FFFFFF !important;
        border: 1px solid #0A2E24 !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 9px 20px !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton>button:hover {
        background-color: #144235 !important;
        border-color: #144235 !important;
        box-shadow: 0 2px 8px rgba(10, 46, 36, 0.18) !important;
    }

    /* Secondary / Download Buttons */
    .stDownloadButton>button {
        background-color: #FFFFFF !important;
        color: #0A2E24 !important;
        border: 1.5px solid #CBD7CF !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
    }

    .stDownloadButton>button:hover {
        background-color: #F1F6F2 !important;
        border-color: #0A2E24 !important;
        color: #06271F !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid #DCE5DE !important;
        padding-bottom: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #4A6357 !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0A2E24 !important;
        background: #E5EDE7 !important;
        border-bottom: 2px solid #0A2E24 !important;
    }

    /* Dataframe Table Contrast */
    [data-testid="stDataFrame"] {
        border: 1px solid #DCE5DE !important;
        border-radius: 8px !important;
    }

    /* Risk Badges */
    .risk-badge-low {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: #D1EAD9;
        color: #083D2A;
        font-weight: 700;
        font-size: 12px;
    }

    .risk-badge-med {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        font-size: 12px;
    }

    .risk-badge-high {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: #FFE4E6;
        color: #9F1239;
        font-weight: 700;
        font-size: 12px;
    }
</style>
"""
