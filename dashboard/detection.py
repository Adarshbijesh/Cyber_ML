"""
detection.py — Dashboard Page 2: Malware Detection
"""

import os
import sys
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASELINE_DIR, SCALER_PATH, FEATURE_NAMES_PATH, SAMPLE_PATH
from src.predict import (
    predict_single, validate_feature_vector, prepare_feature_frame, get_feature_importance,
)
from src.train import list_available_models, load_model


def _load_resources():
    """Load model, scaler, feature names."""
    models = list_available_models()
    if not models:
        return None, None, None, []
    # Default to random_forest
    model_name = "random_forest" if "random_forest" in models else models[0]
    model = load_model(model_name)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    feat_names = []
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            feat_names = json.load(f)
    return model, scaler, model_name, feat_names, models


def render():
    st.markdown("""
    <div style="background:#0F1F3D; border-left:4px solid #00D4FF;
                border-radius:8px; padding:16px 20px; margin-bottom:20px;">
        <h2 style="color:#00D4FF; margin:0; font-size:1.5rem;">🔍 Malware Detection</h2>
        <p style="color:#8899BB; margin:6px 0 0 0; font-size:0.9rem;">
            Upload a feature CSV or select a demo sample to classify as Benign or Malicious.
            The model never executes uploaded files — only feature values are processed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, scaler, model_name, feat_names, all_models = _load_resources()
    except Exception as e:
        st.error(f"Could not load models: {e}. Please run the training pipeline first.")
        return

    if model is None:
        st.warning("No trained models found. Run `python src/train.py` first.")
        return

    # ── Model Selection ───────────────────────────────────────────────────────
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_model_name = st.selectbox(
            "Select Classifier",
            options=all_models,
            index=all_models.index("random_forest") if "random_forest" in all_models else 0,
            key="detection_model_select",
        )
        model = load_model(selected_model_name)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is None:
        expected_features = getattr(scaler, "n_features_in_", len(feat_names))
    expected_features = int(expected_features)

    with col_info:
        st.markdown(f"""
        <div style="background:#162040; border-radius:8px; padding:12px 16px; margin-top:24px;">
            <span style="color:#00D4FF; font-weight:600;">Active Model:</span>
            <span style="color:#E8EAF6;"> {selected_model_name.replace('_',' ').title()}</span><br>
            <span style="color:#8899BB; font-size:0.85rem;">
                Features: {expected_features} | Scaler: {'StandardScaler' if scaler else 'None'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Input Mode ────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input Method",
        ["🎯 Demo Sample", "📂 Upload CSV", "✏️ Manual Entry"],
        horizontal=True,
        key="detection_input_mode",
    )

    sample_vector = None
    apply_scaler = True

    if input_mode == "🎯 Demo Sample":
        if os.path.exists(SAMPLE_PATH):
            sample_df = pd.read_csv(SAMPLE_PATH)
        else:
            st.warning("Sample file not found.")
            return

        label_col = "label" if "label" in sample_df.columns else None
        feat_cols = [c for c in sample_df.columns if c != "label"]

        demo_idx = st.slider("Select sample index", 0, min(len(sample_df)-1, 19), 0, key="demo_slider")
        row = sample_df.iloc[demo_idx]
        true_label = int(row["label"]) if label_col else None
        sample_vector = row[feat_cols].values.astype(float)
        apply_scaler = False

        if true_label is not None:
            label_name = "🟢 Benign" if true_label == 0 else "🔴 Malicious"
            st.markdown(f"""
            <div style="background:#162040; border-radius:6px; padding:10px 14px; display:inline-block;">
                True Label (from dataset): <strong style="color:{'#00E676' if true_label==0 else '#FF1744'};">{label_name}</strong>
            </div>
            """, unsafe_allow_html=True)

    elif input_mode == "📂 Upload CSV":
        st.markdown("""
        <div style="background: rgba(255,215,0,0.08); border: 1px solid rgba(255,215,0,0.3);
                    border-radius:6px; padding:10px 14px; margin-bottom:10px; font-size:0.85rem; color:#FFD600;">
            ⚠️ Upload a CSV with feature columns (no executable files). Feature values only.
        </div>
        """, unsafe_allow_html=True)
        input_is_scaled = st.checkbox(
            "Uploaded values are already standardized",
            value=False,
            help="Enable this for files exported from the project's processed dataset.",
            key="upload_is_scaled",
        )
        apply_scaler = not input_is_scaled
        uploaded = st.file_uploader("Upload feature CSV", type=["csv"], key="upload_csv")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                feature_frame = prepare_feature_frame(df_up, feat_names, expected_features)
                valid, msg = validate_feature_vector(
                    feature_frame.iloc[0].to_numpy(), expected_features
                )
                if not valid:
                    st.error(f"Validation failed: {msg}")
                    return
                sample_vector = feature_frame.iloc[0].to_numpy(dtype=float)
                st.success(f"Loaded {expected_features} features from {uploaded.name}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return

    elif input_mode == "✏️ Manual Entry":
        st.markdown("**Enter feature values** (space or comma separated):")
        default_vals = " ".join(["0.0"] * expected_features)
        raw_input = st.text_area(
            f"Feature values ({expected_features} values)", value=default_vals, height=100,
                                  key="manual_features")
        try:
            vals = [float(x) for x in raw_input.replace(",", " ").split()]
            if len(vals) != expected_features:
                st.warning(f"Expected {expected_features} values, got {len(vals)}")
            else:
                sample_vector = np.array(vals)
        except ValueError:
            st.warning("Invalid input — use numbers separated by spaces")

    # ── Run Prediction ────────────────────────────────────────────────────────
    if sample_vector is not None:
        st.markdown("---")
        predict_btn = st.button("🚀 Run Classification", key="run_predict_btn", use_container_width=True)
        if predict_btn or input_mode == "🎯 Demo Sample":
            with st.spinner("Classifying..."):
                try:
                    result = predict_single(
                        model, sample_vector, scaler if apply_scaler else None
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    return

            # ── Result Card ───────────────────────────────────────────────────
            is_mal = result["label"] == 1
            result_color = "#FF1744" if is_mal else "#00E676"
            result_icon = "🔴 MALICIOUS" if is_mal else "🟢 BENIGN"
            conf_pct = result["confidence"] * 100

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {'#1A0A0A' if is_mal else '#0A1A0A'}, #0F1F3D);
                        border: 2px solid {result_color}; border-radius: 12px; padding: 24px; text-align:center;
                        margin: 16px 0;">
                <div style="font-size: 2rem; font-weight: 800; color: {result_color}; letter-spacing: 2px;">
                    {result_icon}
                </div>
                <div style="font-size: 1.1rem; color: #E8EAF6; margin-top: 8px;">
                    Confidence: <strong style="color:{result_color};">{conf_pct:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence gauge
            r_col, l_col = st.columns(2)
            with r_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf_pct,
                    number={"suffix": "%", "font": {"color": result_color, "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8899BB"},
                        "bar": {"color": result_color},
                        "bgcolor": "#162040",
                        "borderwidth": 0,
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.8,
                            "value": 50,
                        },
                        "steps": [
                            {"range": [0, 50], "color": "#0F1F3D"},
                            {"range": [50, 100], "color": "#162040"},
                        ],
                    },
                    title={"text": "Classification Confidence", "font": {"color": "#8899BB"}},
                ))
                fig_gauge.update_layout(
                    height=240, paper_bgcolor="#0A1628", font=dict(color="#E8EAF6"),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with l_col:
                # Probability bars
                probs = result["probabilities"]
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    y=["Benign", "Malicious"],
                    x=[probs["benign"] * 100, probs["malicious"] * 100],
                    orientation="h",
                    marker_color=["#00E676", "#FF1744"],
                    text=[f"{probs['benign']*100:.1f}%", f"{probs['malicious']*100:.1f}%"],
                    textposition="outside",
                ))
                fig_prob.update_layout(
                    title="Class Probabilities",
                    xaxis=dict(range=[0, 115], title="Probability (%)", gridcolor="#1A2744"),
                    yaxis=dict(gridcolor="#1A2744"),
                    height=240,
                    paper_bgcolor="#0A1628",
                    plot_bgcolor="#0F1F3D",
                    font=dict(color="#E8EAF6"),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            # ── Feature Importance ─────────────────────────────────────────────
            if feat_names:
                st.markdown("#### 🔎 Top Feature Contributions")
                try:
                    from src.visualization import plot_feature_importance
                    imp_df = get_feature_importance(model, feat_names)
                    if not imp_df.empty:
                        fig_imp = plot_feature_importance(imp_df, top_n=15)
                        st.plotly_chart(fig_imp, use_container_width=True)
                except Exception:
                    pass

            # ── Feature Values Table ───────────────────────────────────────────
            with st.expander("📋 Feature Values", expanded=False):
                disp_names = feat_names if feat_names else [f"feature_{i}" for i in range(N_FEATURES)]
                df_feat = pd.DataFrame({
                    "Feature": disp_names[:len(sample_vector)],
                    "Value": sample_vector.tolist(),
                })
                st.dataframe(df_feat, use_container_width=True, hide_index=True)
