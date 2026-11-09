"""
evaluation.py (dashboard) — Dashboard Page 5: Model Evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_DIR, DEFENDED_DIR, RESULTS_DIR, TEST_PATH, SCALER_PATH,
    FEATURE_NAMES_PATH, PGD_EPSILON, N_FEATURES,
)
from src.train import load_model, list_available_models
from src.data_loader import split_features_labels
from src.evaluation import compute_full_metrics, compute_roc_data
from src.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_metrics_radar,
    plot_attack_comparison, apply_template,
)


def render():
    st.markdown("""
    <div style="background:#0F1F3D; border-left:4px solid #FFD600;
                border-radius:8px; padding:16px 20px; margin-bottom:20px;">
        <h2 style="color:#FFD600; margin:0; font-size:1.5rem;">📊 Model Evaluation</h2>
        <p style="color:#8899BB; margin:6px 0 0 0; font-size:0.9rem;">
            Comprehensive performance metrics: confusion matrices, ROC curves, robustness comparison.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    if not os.path.exists(TEST_PATH):
        st.warning("Test data not found. Run preprocessing first.")
        return

    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test, feat_names = split_features_labels(test_df)

    baseline_models = list_available_models()
    if not baseline_models:
        st.warning("No trained models found.")
        return

    # ── Model selector ─────────────────────────────────────────────────────────
    st.markdown("#### 🎛️ Select Models to Evaluate")
    all_model_options = baseline_models.copy()

    # Add defended models
    defended_options = {}
    if os.path.exists(DEFENDED_DIR):
        for f in os.listdir(DEFENDED_DIR):
            if f.endswith(".joblib"):
                name = f.replace(".joblib", "")
                defended_options[name] = os.path.join(DEFENDED_DIR, f)
                all_model_options.append(f"[Defended] {name}")

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        selected_models = st.multiselect(
            "Models to compare",
            options=all_model_options,
            default=baseline_models[:min(3, len(baseline_models))],
            key="eval_model_select",
        )
    with sel_col2:
        show_defended = st.checkbox("Include defended models", value=True, key="show_defended")

    if not selected_models:
        st.info("Select at least one model above.")
        return

    # ── Compute metrics for all selected models ────────────────────────────────
    model_metrics = {}
    roc_data_all = {}

    for model_name in selected_models:
        try:
            if model_name.startswith("[Defended] "):
                clean_name = model_name.replace("[Defended] ", "")
                model_path = defended_options.get(clean_name)
                if model_path and os.path.exists(model_path):
                    model = joblib.load(model_path)
                else:
                    continue
            else:
                model = load_model(model_name)

            preds = model.predict(X_test)
            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]

            metrics = compute_full_metrics(y_test, preds, probs, label=model_name)
            model_metrics[model_name] = metrics
            roc_data_all[model_name] = compute_roc_data(model, X_test, y_test)
        except Exception as e:
            st.warning(f"Could not evaluate {model_name}: {e}")

    if not model_metrics:
        st.error("No models could be evaluated.")
        return

    # ── Summary Metrics Table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Performance Summary")

    rows = []
    for name, m in model_metrics.items():
        rows.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A",
            "FPR": f"{m['false_positive_rate']:.4f}",
            "FNR": f"{m['false_negative_rate']:.4f}",
        })
    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ── Export Buttons ────────────────────────────────────────────────────────
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_str = df_summary.to_csv(index=False)
        st.download_button(
            "⬇️ Export CSV",
            data=csv_str,
            file_name="evaluation_metrics.csv",
            mime="text/csv",
            key="export_csv_btn",
        )
    with exp_col2:
        json_str = json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in model_metrics.items()},
            indent=2, default=str
        )
        st.download_button(
            "⬇️ Export JSON",
            data=json_str,
            file_name="evaluation_metrics.json",
            mime="application/json",
            key="export_json_btn",
        )

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔲 Confusion Matrices")

    cm_cols = st.columns(min(3, len(model_metrics)))
    for col, (name, m) in zip(cm_cols, model_metrics.items()):
        with col:
            cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
            fig_cm = plot_confusion_matrix(cm, name.replace("_", " ").title()[:20])
            fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curves ───────────────────────────────────────────────────────────
    if any(roc_data_all.values()):
        st.markdown("---")
        st.markdown("#### 📈 ROC Curves")
        fig_roc = plot_roc_curves(roc_data_all)
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Radar Chart ───────────────────────────────────────────────────────────
    if len(model_metrics) > 1:
        st.markdown("---")
        st.markdown("#### 🕸️ Performance Radar Chart")
        fig_radar = plot_metrics_radar(model_metrics)
        fig_radar.update_layout(height=440)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Metric Bar Charts ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Metric Comparison")
    metric_tab = st.tabs(["Accuracy", "F1-Score", "ROC-AUC", "FPR vs FNR"])

    model_names_disp = [n.replace("_", " ").title() for n in model_metrics.keys()]

    with metric_tab[0]:
        vals = [m["accuracy"] * 100 for m in model_metrics.values()]
        fig = go.Figure(go.Bar(x=model_names_disp, y=vals,
                                marker_color="#00D4FF",
                                text=[f"{v:.2f}%" for v in vals],
                                textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 115], title="Accuracy (%)"),
                           paper_bgcolor="#0A1628", plot_bgcolor="#0F1F3D",
                           font=dict(color="#E8EAF6"), height=320,
                           margin=dict(l=20, r=20, t=20, b=60),
                           xaxis=dict(tickangle=-15))
        st.plotly_chart(fig, use_container_width=True)

    with metric_tab[1]:
        vals = [m["f1"] * 100 for m in model_metrics.values()]
        fig = go.Figure(go.Bar(x=model_names_disp, y=vals,
                                marker_color="#00E676",
                                text=[f"{v:.2f}%" for v in vals],
                                textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 115], title="F1-Score (%)"),
                           paper_bgcolor="#0A1628", plot_bgcolor="#0F1F3D",
                           font=dict(color="#E8EAF6"), height=320,
                           margin=dict(l=20, r=20, t=20, b=60),
                           xaxis=dict(tickangle=-15))
        st.plotly_chart(fig, use_container_width=True)

    with metric_tab[2]:
        vals = [(m.get("roc_auc") or 0) * 100 for m in model_metrics.values()]
        fig = go.Figure(go.Bar(x=model_names_disp, y=vals,
                                marker_color="#7B2FBE",
                                text=[f"{v:.2f}%" for v in vals],
                                textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 115], title="ROC-AUC (%)"),
                           paper_bgcolor="#0A1628", plot_bgcolor="#0F1F3D",
                           font=dict(color="#E8EAF6"), height=320,
                           margin=dict(l=20, r=20, t=20, b=60),
                           xaxis=dict(tickangle=-15))
        st.plotly_chart(fig, use_container_width=True)

    with metric_tab[3]:
        fprs = [m["false_positive_rate"] * 100 for m in model_metrics.values()]
        fnrs = [m["false_negative_rate"] * 100 for m in model_metrics.values()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="FPR", x=model_names_disp, y=fprs,
                              marker_color="#FFD600",
                              text=[f"{v:.2f}%" for v in fprs],
                              textposition="outside"))
        fig.add_trace(go.Bar(name="FNR", x=model_names_disp, y=fnrs,
                              marker_color="#FF6B35",
                              text=[f"{v:.2f}%" for v in fnrs],
                              textposition="outside"))
        fig.update_layout(barmode="group",
                           yaxis=dict(range=[0, 50], title="Rate (%)"),
                           paper_bgcolor="#0A1628", plot_bgcolor="#0F1F3D",
                           font=dict(color="#E8EAF6"), height=320,
                           margin=dict(l=20, r=20, t=20, b=60),
                           xaxis=dict(tickangle=-15))
        st.plotly_chart(fig, use_container_width=True)

    # ── Attack Results Summary ─────────────────────────────────────────────────
    attack_csv = os.path.join(RESULTS_DIR, "attack_results.csv")
    if os.path.exists(attack_csv):
        st.markdown("---")
        st.markdown("#### ⚔️ Attack Experiment Results")
        df_atk = pd.read_csv(attack_csv)
        st.dataframe(df_atk, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Attack Results CSV",
            data=df_atk.to_csv(index=False),
            file_name="attack_results.csv",
            mime="text/csv",
            key="dl_attack_csv",
        )
