"""
app.py — MalwareShield AI: Adversarial Robustness Laboratory
Main Streamlit entry point.

Launch: streamlit run app.py
"""

import os
import sys
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MalwareShield AI — Adversarial Robustness Laboratory",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/elastic/ember",
        "About": "MalwareShield AI — Academic adversarial ML research platform",
    },
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A1628 !important;
    font-family: 'Inter', sans-serif !important;
    color: #E8EAF6 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B30 0%, #0A1628 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #8899BB !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: none !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(0,212,255,0.08) !important;
}

/* ── Main content area ── */
[data-testid="stMain"] {
    background-color: #0A1628 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0066CC, #004499) !important;
    color: white !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0088FF, #0055CC) !important;
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.3) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0F1F3D !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 8px !important;
    color: #E8EAF6 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #00D4FF !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: #0F1F3D !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0F1F3D !important;
    border-radius: 8px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8899BB !important;
    background: transparent !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    background: rgba(0,212,255,0.1) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0F1F3D !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8EAF6 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0F1F3D !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* ── Info/Warning/Error boxes ── */
.stAlert {
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}

/* ── Code blocks ── */
.stCode {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0D1B30 !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A1628; }
::-webkit-scrollbar-thumb { background: #1A2744; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00D4FF44; }

/* ── HR ── */
hr { border-color: rgba(0,212,255,0.1) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(0,212,255,0.08) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
}
.stDownloadButton > button:hover {
    background: rgba(0,212,255,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0F1F3D !important;
    border: 1px dashed rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
}

/* ── Multiselect ── */
.stMultiSelect > div > div {
    background: #0F1F3D !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
}

/* ── Radio ── */
.stRadio > div { gap: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center; padding:20px 10px 16px 10px; border-bottom:1px solid rgba(0,212,255,0.1); margin-bottom:16px;">
        <div style="font-size:2.5rem; margin-bottom:4px;">🛡️</div>
        <div style="color:#00D4FF; font-weight:800; font-size:1.1rem; letter-spacing:-0.3px;">MalwareShield AI</div>
        <div style="color:#8899BB; font-size:0.75rem; margin-top:2px;">Adversarial Robustness Lab</div>
        <div style="margin-top:10px; padding:4px 10px; background:rgba(255,215,0,0.08);
                    border:1px solid rgba(255,215,0,0.25); border-radius:20px;
                    color:#FFD600; font-size:0.7rem;">RESEARCH / DEMO MODE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔍 Malware Detection",
            "⚔️ Adversarial Attack Lab",
            "🛡️ Defense Laboratory",
            "📊 Model Evaluation",
            "📚 Research & Docs",
        ],
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Quick status panel
    from config import BASELINE_DIR, DEFENDED_DIR, RESULTS_DIR
    models_ready = os.path.exists(BASELINE_DIR) and any(
        f.endswith(".joblib") for f in os.listdir(BASELINE_DIR)
    ) if os.path.exists(BASELINE_DIR) else False
    defended_ready = os.path.exists(DEFENDED_DIR) and any(
        f.endswith(".joblib") for f in os.listdir(DEFENDED_DIR)
    ) if os.path.exists(DEFENDED_DIR) else False
    results_ready = os.path.exists(RESULTS_DIR) and len(os.listdir(RESULTS_DIR)) > 0

    def status_dot(ready):
        return f'<span style="color:{"#00E676" if ready else "#FF6B35"}; font-size:0.9rem;">{"●" if ready else "○"}</span>'

    st.markdown(f"""
    <div style="background:#0F1F3D; border-radius:8px; padding:12px 14px;
                border:1px solid rgba(0,212,255,0.08);">
        <div style="color:#8899BB; font-size:0.75rem; font-weight:600; letter-spacing:0.5px;
                    text-transform:uppercase; margin-bottom:8px;">System Status</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin:4px 0;">
            {status_dot(True)}
            <span style="color:#E8EAF6; font-size:0.82rem; margin-left:6px; flex:1;">Dataset</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin:4px 0;">
            {status_dot(models_ready)}
            <span style="color:#E8EAF6; font-size:0.82rem; margin-left:6px; flex:1;">Baseline Models</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin:4px 0;">
            {status_dot(defended_ready)}
            <span style="color:#E8EAF6; font-size:0.82rem; margin-left:6px; flex:1;">Defense Models</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin:4px 0;">
            {status_dot(results_ready)}
            <span style="color:#E8EAF6; font-size:0.82rem; margin-left:6px; flex:1;">Experiment Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Quick commands
    with st.expander("⚡ Quick Commands"):
        st.code("# 1. Run full pipeline\npython experiments/run_experiments.py", language="bash")
        st.code("# 2. Train only\npython src/train.py", language="bash")
        st.code("# 3. Launch app\nstreamlit run app.py", language="bash")

    st.markdown("""
    <div style="margin-top:auto; padding:12px 0 4px 0; text-align:center;
                color:#8899BB; font-size:0.72rem; border-top:1px solid rgba(0,212,255,0.08);">
        Academic Research Project<br>
        Python 3.11 · Scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Page Router ───────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    from dashboard import overview
    overview.render()

elif page == "🔍 Malware Detection":
    from dashboard import detection
    detection.render()

elif page == "⚔️ Adversarial Attack Lab":
    from dashboard import attack_lab
    attack_lab.render()

elif page == "🛡️ Defense Laboratory":
    from dashboard import defense_lab
    defense_lab.render()

elif page == "📊 Model Evaluation":
    from dashboard import evaluation
    evaluation.render()

elif page == "📚 Research & Docs":
    from dashboard import research
    research.render()
