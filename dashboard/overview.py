"""
overview.py — Dashboard Page 1: Project Overview
"""

import os
import sys
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, PROCESSED_DIR, BASELINE_DIR, DEFENDED_DIR


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0A1628 0%, #0F1F3D 50%, #162040 100%);
                border: 1px solid rgba(0,212,255,0.2); border-radius: 12px; padding: 24px; margin-bottom: 24px;">
        <h1 style="color: #00D4FF; font-size: 2rem; margin: 0; letter-spacing: -0.5px;">
            🛡️ MalwareShield AI
        </h1>
        <p style="color: #7CB9E8; font-size: 1.1rem; margin: 8px 0 0 0; font-weight: 300;">
            Adversarial Robustness Laboratory — Academic Research Platform
        </p>
        <div style="margin-top: 12px; padding: 8px 14px; background: rgba(255,215,0,0.1);
                    border-left: 3px solid #FFD600; border-radius: 4px; font-size: 0.85rem; color: #FFD600;">
            ⚠️ RESEARCH MODE: Uses a synthetic demonstration dataset. Not real malware data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # Load dataset stats
    dataset_path = os.path.join(PROCESSED_DIR, "demo_dataset.csv")
    n_total, n_benign, n_malicious = 10000, 6000, 4000
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path, usecols=["label"])
            n_total = len(df)
            n_malicious = int(df["label"].sum())
            n_benign = n_total - n_malicious
        except Exception:
            pass

    n_models = len([f for f in os.listdir(BASELINE_DIR)
                    if f.endswith(".joblib")]) if os.path.exists(BASELINE_DIR) else 0
    n_defended = len([f for f in os.listdir(DEFENDED_DIR)
                      if f.endswith(".joblib")]) if os.path.exists(DEFENDED_DIR) else 0

    metric_cards = [
        (col1, "Total Samples", f"{n_total:,}", "#00D4FF"),
        (col2, "Benign Files", f"{n_benign:,}", "#00E676"),
        (col3, "Malicious Files", f"{n_malicious:,}", "#FF1744"),
        (col4, "Trained Models", f"{n_models + n_defended}", "#7B2FBE"),
    ]
    for col, label, value, color in metric_cards:
        with col:
            st.markdown(f"""
            <div style="background: #0F1F3D; border: 1px solid {color}33;
                        border-top: 3px solid {color}; border-radius: 10px;
                        padding: 16px; text-align: center;">
                <div style="color: {color}; font-size: 1.8rem; font-weight: 700;">{value}</div>
                <div style="color: #8899BB; font-size: 0.85rem; margin-top: 4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Two column layout ──────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### 📌 Research Objectives")
        objectives = [
            ("🔍 Malware Detection", "Train ML classifiers to distinguish benign vs malicious PE features"),
            ("⚔️ Adversarial Attacks", "Simulate FGSM, PGD, Genetic Algorithm, and Random attacks in feature space"),
            ("🛡️ Defense Mechanisms", "Implement adversarial training, input validation, and ensemble defense"),
            ("📊 Robustness Analysis", "Quantify accuracy degradation and attack success rates"),
            ("🎓 Educational Tool", "Demonstrate the ML security lifecycle for academic evaluation"),
        ]
        for icon_title, desc in objectives:
            st.markdown(f"""
            <div style="background: #0F1F3D; border-left: 3px solid #00D4FF;
                        border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <strong style="color: #E8EAF6;">{icon_title}</strong>
                <div style="color: #8899BB; font-size: 0.85rem; margin-top: 2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("#### 🏗️ System Architecture")
        # Architecture diagram using Plotly
        fig = go.Figure()
        nodes = [
            ("Dataset\n(Synthetic PE)", 0.5, 0.9, "#00D4FF"),
            ("Preprocessing", 0.5, 0.75, "#7CB9E8"),
            ("Baseline Models\n(RF / GB / LR)", 0.5, 0.58, "#00E676"),
            ("Adversarial\nAttacks", 0.2, 0.4, "#FF6B35"),
            ("Defense\nModels", 0.8, 0.4, "#7B2FBE"),
            ("Robustness\nEvaluation", 0.5, 0.2, "#FFD600"),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)]

        xs = [n[1] for n in nodes]
        ys = [n[2] for n in nodes]
        colors = [n[3] for n in nodes]
        labels = [n[0] for n in nodes]

        for src, dst in edges:
            fig.add_trace(go.Scatter(
                x=[xs[src], xs[dst]], y=[ys[src], ys[dst]],
                mode="lines",
                line=dict(color="rgba(0,212,255,0.3)", width=2),
                showlegend=False,
                hoverinfo="none",
            ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=20, color=colors, opacity=0.9),
            text=labels,
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            showlegend=False,
            hoverinfo="text",
        ))
        fig.update_layout(
            height=320,
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            xaxis=dict(visible=False, range=[-0.1, 1.1]),
            yaxis=dict(visible=False, range=[0.05, 1.05]),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Dataset Information ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📁 Dataset Information")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("""
        <div style="background: #0F1F3D; border-radius: 10px; padding: 16px;">
            <h4 style="color: #00D4FF; margin-top: 0;">Synthetic Demo Dataset</h4>
            <table style="width: 100%; color: #E8EAF6; font-size: 0.9rem;">
                <tr><td style="color:#8899BB; padding: 4px 0;">Source</td><td>Synthetically generated (PE-inspired)</td></tr>
                <tr><td style="color:#8899BB; padding: 4px 0;">Total Samples</td><td>10,000</td></tr>
                <tr><td style="color:#8899BB; padding: 4px 0;">Features</td><td>50 (PE headers, entropy, imports, strings)</td></tr>
                <tr><td style="color:#8899BB; padding: 4px 0;">Train / Val / Test</td><td>70% / 15% / 15%</td></tr>
                <tr><td style="color:#8899BB; padding: 4px 0;">Class Ratio</td><td>60% Benign / 40% Malicious</td></tr>
                <tr><td style="color:#8899BB; padding: 4px 0;">Preprocessing</td><td>StandardScaler (fit on train only)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with dcol2:
        # Class distribution pie
        fig_pie = go.Figure(go.Pie(
            labels=["Benign", "Malicious"],
            values=[n_benign, n_malicious],
            hole=0.5,
            marker=dict(colors=["#00E676", "#FF1744"]),
            textinfo="label+percent",
            textfont=dict(color="white", size=12),
        ))
        fig_pie.update_layout(
            height=220,
            paper_bgcolor="#0F1F3D",
            plot_bgcolor="#0F1F3D",
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(color="#E8EAF6"),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Model Performance Overview ────────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    if os.path.exists(metrics_path):
        st.markdown("---")
        st.markdown("#### 🤖 Baseline Model Performance")
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            best = metrics.get("best_model", "")
            rows = []
            for name in ["random_forest", "gradient_boosting", "logistic_regression"]:
                if name in metrics:
                    m = metrics[name].get("test", {})
                    rows.append({
                        "Model": name.replace("_", " ").title() + (" ⭐ Best" if name == best else ""),
                        "Accuracy": f"{m.get('accuracy', 0):.4f}",
                        "Precision": f"{m.get('precision', 0):.4f}",
                        "Recall": f"{m.get('recall', 0):.4f}",
                        "F1-Score": f"{m.get('f1', 0):.4f}",
                        "ROC-AUC": f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "N/A",
                    })
            if rows:
                df_metrics = pd.DataFrame(rows)
                st.dataframe(
                    df_metrics,
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as e:
            st.info(f"Model metrics not yet available. Run experiments first.")
    else:
        st.info("🔄 Model metrics will appear here after training completes. Use the sidebar to navigate to Adversarial Attack Lab after training.")

    # ── Technology Stack ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧰 Technology Stack")
    tech_cols = st.columns(5)
    techs = [
        ("🐍 Python 3.11", "Runtime"),
        ("🤖 Scikit-learn", "ML Framework"),
        ("📊 XGBoost", "Gradient Boosting"),
        ("🌊 Streamlit", "Dashboard"),
        ("📈 Plotly", "Visualization"),
    ]
    for col, (name, role) in zip(tech_cols, techs):
        with col:
            st.markdown(f"""
            <div style="background: #0F1F3D; border-radius: 8px; padding: 10px; text-align: center;
                        border: 1px solid rgba(0,212,255,0.15);">
                <div style="font-size: 1.3rem;">{name.split()[0]}</div>
                <div style="color: #00D4FF; font-size: 0.8rem; font-weight: 600;">{' '.join(name.split()[1:])}</div>
                <div style="color: #8899BB; font-size: 0.75rem;">{role}</div>
            </div>
            """, unsafe_allow_html=True)
