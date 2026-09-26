"""
Stitch Design System Tokens & CSS for SynthGuard.
Theme: 'Calm Editorial Intelligence'
Primary: Deep Juniper (#06271F / #1E3D34)
Secondary: Sage Slate (#3A6753 / #BAEBD1)
Canvas: Alabaster (#F6FBF5 / #FFFFFF)
Fonts: Source Serif 4, Inter, JetBrains Mono
"""

STITCH_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
    /* Global Canvas & Base Styling */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F6FBF5 !important;
        color: #181D1A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #F0F5F0 !important;
        border-right: 1px solid #DFE4DF !important;
    }

    /* Headings in Source Serif 4 */
    h1, h2, h3, .stitch-title {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #06271F !important;
        font-weight: 500 !important;
        letter-spacing: -0.015em !important;
    }

    .stitch-display {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 2.25rem !important;
        line-height: 1.25 !important;
        font-weight: 400 !important;
        color: #06271F !important;
        margin-bottom: 0.5rem !important;
    }

    .stitch-subhead {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.05rem !important;
        color: #414845 !important;
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
        background: rgba(246, 251, 245, 0.95);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #DFE4DF;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 1px 4px rgba(6, 39, 31, 0.04);
    }

    .stitch-badge-ready {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        background-color: #BAEBD1;
        color: #002115;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .stitch-badge-ready::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #3A6753;
    }

    /* Stepper Bar */
    .stitch-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding-top: 8px;
        overflow-x: auto;
    }

    .stitch-step-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        color: #414845;
        text-decoration: none;
    }

    .stitch-step-item.active {
        color: #06271F;
        font-weight: 600;
    }

    .stitch-step-num {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #E5E9E4;
        color: #414845;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
    }

    .stitch-step-item.active .stitch-step-num {
        background: #06271F;
        color: #FFFFFF;
    }

    .stitch-step-arrow {
        color: #C1C8C4;
        font-size: 12px;
    }

    /* Surface Cards */
    .stitch-card {
        background: #FFFFFF;
        border: 1px solid #DFE4DF;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(6, 39, 31, 0.02);
    }

    .stitch-card-inset {
        background: #F0F5F0;
        border: 1px solid #DFE4DF;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #06271F !important;
        color: #FFFFFF !important;
        border: 1px solid #06271F !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton>button:hover {
        background-color: #1E3D34 !important;
        border-color: #1E3D34 !important;
        box-shadow: 0 2px 6px rgba(6, 39, 31, 0.15) !important;
    }

    /* Secondary / Download Buttons */
    .stDownloadButton>button {
        background-color: #FFFFFF !important;
        color: #06271F !important;
        border: 1px solid #DFE4DF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    .stDownloadButton>button:hover {
        background-color: #F0F5F0 !important;
        border-color: #C1C8C4 !important;
    }

    /* Risk Badges */
    .risk-badge-low {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        background-color: #BAEBD1;
        color: #002115;
        font-weight: 600;
        font-size: 12px;
    }

    .risk-badge-med {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        background-color: #FFDCBF;
        color: #542E00;
        font-weight: 600;
        font-size: 12px;
    }

    .risk-badge-high {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        background-color: #FFDAD6;
        color: #93000A;
        font-weight: 600;
        font-size: 12px;
    }
</style>
"""
