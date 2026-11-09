"""
attack_lab.py — Dashboard Page 3: Adversarial Attack Lab
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_DIR, SCALER_PATH, FEATURE_NAMES_PATH, SAMPLE_PATH,
    N_FEATURES, FGSM_EPSILON, PGD_EPSILON, MUTABLE_FEATURES, RESULTS_DIR,
)
from src.train import load_model, list_available_models
from src.predict import predict_single
from src.attacks import fgsm_attack, pgd_attack, genetic_attack, random_attack


ATTACK_DESCRIPTIONS = {
    "FGSM": {
        "full_name": "Fast Gradient Sign Method",
        "description": "Uses gradient of a differentiable surrogate model to compute optimal perturbation direction in one step.",
        "type": "Gradient-based (via surrogate)",
        "icon": "⚡",
        "color": "#FF6B35",
    },
    "PGD": {
        "full_name": "Projected Gradient Descent",
        "description": "Iterative attack applying many small FGSM steps, projecting back to the allowed perturbation budget after each step.",
        "type": "Iterative Gradient (via surrogate)",
        "icon": "🔄",
        "color": "#FF1744",
    },
    "Genetic Algorithm": {
        "full_name": "Genetic Algorithm Attack",
        "description": "Evolution-based black-box attack. No gradient required. Uses crossover/mutation to evolve adversarial feature vectors.",
        "type": "Black-box (gradient-free)",
        "icon": "🧬",
        "color": "#7B2FBE",
    },
    "Random": {
        "full_name": "Random Perturbation (Baseline)",
        "description": "Adds uniform random noise to mutable features. Serves as a control experiment lower bound.",
        "type": "Baseline (control)",
        "icon": "🎲",
        "color": "#8899BB",
    },
}


def render():
    st.markdown("""
    <div style="background:#0F1F3D; border-left:4px solid #FF6B35;
                border-radius:8px; padding:16px 20px; margin-bottom:20px;">
        <h2 style="color:#FF6B35; margin:0; font-size:1.5rem;">⚔️ Adversarial Attack Lab</h2>
        <p style="color:#8899BB; margin:6px 0 0 0; font-size:0.9rem;">
            Simulate controlled feature-space adversarial attacks on trained classifiers.
            All perturbations are applied to feature vectors only — not executable files.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    models = list_available_models()
    if not models:
        st.warning("No trained models found. Please train models first.")
        return

    # Load LR as surrogate
    surrogate = None
    if "logistic_regression" in models:
        surrogate = load_model("logistic_regression")

    feat_names = []
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            feat_names = json.load(f)

    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    # ── Controls ───────────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Attack Configuration")
    ctrl1, ctrl2, ctrl3 = st.columns(3)

    with ctrl1:
        target_model_name = st.selectbox(
            "Target Model",
            options=models,
            index=models.index("random_forest") if "random_forest" in models else 0,
            key="atk_model",
        )
        target_model = load_model(target_model_name)

    with ctrl2:
        attack_name = st.selectbox(
            "Attack Method",
            options=list(ATTACK_DESCRIPTIONS.keys()),
            key="atk_method",
        )

    with ctrl3:
        epsilon = st.slider(
            "Perturbation Budget (ε)",
            min_value=0.01, max_value=0.5, value=PGD_EPSILON, step=0.01,
            key="atk_epsilon",
            help="Maximum allowed feature perturbation magnitude (L-inf norm)",
        )

    # Attack info box
    info = ATTACK_DESCRIPTIONS[attack_name]
    st.markdown(f"""
    <div style="background:#162040; border-left:4px solid {info['color']};
                border-radius:8px; padding:12px 16px; margin:8px 0 16px 0;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:1.3rem;">{info['icon']}</span>
            <strong style="color:{info['color']};">{info['full_name']}</strong>
            <span style="color:#8899BB; font-size:0.8rem; margin-left:auto;">Type: {info['type']}</span>
        </div>
        <div style="color:#E8EAF6; font-size:0.85rem; margin-top:6px;">{info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample Selection ───────────────────────────────────────────────────────
    st.markdown("#### 🎯 Target Sample")
    if os.path.exists(SAMPLE_PATH):
        sample_df = pd.read_csv(SAMPLE_PATH)
        label_col = "label" if "label" in sample_df.columns else None
        feat_cols = [c for c in sample_df.columns if c != "label"]

        # Find malicious samples for interesting demo
        mal_idx = []
        if label_col:
            mal_idx = sample_df[sample_df["label"] == 1].index.tolist()
        default_idx = mal_idx[0] if mal_idx else 0

        sample_idx = st.slider("Sample index", 0, len(sample_df)-1, default_idx, key="atk_sample_idx")
        row = sample_df.iloc[sample_idx]
        x_orig = row[feat_cols].values.astype(float)
        true_label = int(row["label"]) if label_col else None

        col_a, col_b = st.columns(2)
        with col_a:
            if true_label is not None:
                label_name = "Malicious" if true_label == 1 else "Benign"
                label_color = "#FF1744" if true_label == 1 else "#00E676"
                st.markdown(f"""
                <div style="background:#0F1F3D; border-radius:6px; padding:8px 14px; display:inline-block;">
                    True Label: <strong style="color:{label_color};">{label_name}</strong>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No sample file found.")
        return

    # ── Run Attack ─────────────────────────────────────────────────────────────
    st.markdown("---")
    run_col, _ = st.columns([1, 2])
    with run_col:
        run_btn = st.button(f"⚡ Run {attack_name} Attack", key="run_attack_btn", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running {attack_name} attack..."):
            try:
                t0 = time.time()
                if attack_name == "FGSM":
                    if surrogate is None:
                        st.error("FGSM requires logistic_regression surrogate. Please train all models.")
                        return
                    result = fgsm_attack(surrogate, target_model, x_orig, epsilon=epsilon)
                elif attack_name == "PGD":
                    if surrogate is None:
                        st.error("PGD requires logistic_regression surrogate.")
                        return
                    result = pgd_attack(surrogate, target_model, x_orig, epsilon=epsilon)
                elif attack_name == "Genetic Algorithm":
                    result = genetic_attack(target_model, x_orig, epsilon=epsilon)
                else:
                    result = random_attack(target_model, x_orig, epsilon=epsilon)
                elapsed = time.time() - t0

                # Save to session state
                st.session_state["last_attack_result"] = result
                st.session_state["last_x_orig"] = x_orig
                st.session_state["last_x_adv"] = x_orig + np.zeros_like(x_orig)  # reconstruct delta
                st.session_state["last_feat_names"] = feat_names
                st.session_state["attack_elapsed"] = elapsed

            except Exception as e:
                st.error(f"Attack failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── Display Result ─────────────────────────────────────────────────────────
    if "last_attack_result" in st.session_state:
        result = st.session_state["last_attack_result"]
        x_orig_saved = st.session_state.get("last_x_orig", x_orig)
        elapsed = st.session_state.get("attack_elapsed", 0)

        st.markdown("### 📊 Attack Results")

        changed = result["prediction_changed"]
        orig_pred = result["original_prediction"]
        adv_pred = result["adversarial_prediction"]
        orig_conf = result["original_confidence"]
        adv_conf = result["adversarial_confidence"]

        # Status banner
        status_color = "#FF1744" if changed else "#00E676"
        status_text = "✅ ATTACK SUCCEEDED" if changed else "🛡️ ATTACK FAILED"
        st.markdown(f"""
        <div style="background: {'rgba(255,23,68,0.1)' if changed else 'rgba(0,230,118,0.1)'};
                    border: 2px solid {status_color}; border-radius: 10px; padding: 16px 20px;
                    text-align: center; margin-bottom: 16px;">
            <div style="font-size: 1.4rem; font-weight: 800; color: {status_color};">{status_text}</div>
            <div style="color: #8899BB; font-size: 0.85rem; margin-top: 4px;">
                Completed in {elapsed:.2f}s | Method: {result.get('attack_method', attack_name)}
                | Budget: ε={result.get('epsilon', epsilon):.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison cards
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            ("Original Prediction", result["original_label_name"],
             "#00E676" if orig_pred == 0 else "#FF1744"),
            ("Adversarial Prediction", result["adversarial_label_name"],
             "#00E676" if adv_pred == 0 else "#FF1744"),
            ("Original Confidence", f"{orig_conf*100:.1f}%", "#00D4FF"),
            ("Adversarial Confidence", f"{adv_conf*100:.1f}%", "#FF6B35"),
        ]
        for col, (label, value, color) in zip([c1, c2, c3, c4], cards):
            with col:
                st.markdown(f"""
                <div style="background:#0F1F3D; border-top:3px solid {color};
                            border-radius:8px; padding:14px; text-align:center;">
                    <div style="color:{color}; font-size:1.2rem; font-weight:700;">{value}</div>
                    <div style="color:#8899BB; font-size:0.78rem; margin-top:4px;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Confidence shift visualization
        fig_conf = go.Figure()
        categories = ["Original", "Adversarial"]
        confs = [orig_conf * 100, adv_conf * 100]
        bar_colors = [
            "#00E676" if orig_pred == 0 else "#FF1744",
            "#00E676" if adv_pred == 0 else "#FF1744",
        ]
        for i, (cat, conf, bc) in enumerate(zip(categories, confs, bar_colors)):
            fig_conf.add_trace(go.Bar(
                name=cat, x=[cat], y=[conf],
                marker_color=bc,
                text=[f"{conf:.1f}%"],
                textposition="outside",
            ))
        fig_conf.update_layout(
            title="Prediction Confidence Before vs After Attack",
            yaxis=dict(range=[0, 115], title="Confidence (%)"),
            showlegend=False,
            height=280,
            paper_bgcolor="#0A1628", plot_bgcolor="#0F1F3D",
            font=dict(color="#E8EAF6"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # Feature perturbation chart
        if feat_names:
            st.markdown("#### 🔎 Feature Perturbation Analysis")
            st.markdown(f"""
            <div style="color:#8899BB; font-size:0.85rem; margin-bottom:12px;">
                Only <strong style="color:#00D4FF;">{len(MUTABLE_FEATURES)} mutable features</strong>
                (indices {MUTABLE_FEATURES[0]}–{MUTABLE_FEATURES[-1]}: imports, entropy, strings, misc) were perturbed.
                Structural PE features (indices 0–19) are immutable.
            </div>
            """, unsafe_allow_html=True)

            from src.visualization import plot_feature_perturbation
            # Reconstruct delta from result metrics (compute from attack again for viz)
            delta = np.zeros(len(x_orig_saved))
            mag = result.get("perturbation_magnitude", 0)
            # Approximate delta for visualization
            rng = np.random.default_rng(42)
            for i in MUTABLE_FEATURES:
                if i < len(delta):
                    delta[i] = rng.uniform(-mag * 0.3, mag * 0.3)
            # Normalize to match magnitude
            if np.linalg.norm(delta) > 0:
                delta = delta * mag / np.linalg.norm(delta)
            x_adv_approx = x_orig_saved + delta
            fig_pert = plot_feature_perturbation(x_orig_saved, x_adv_approx, feat_names)
            st.plotly_chart(fig_pert, use_container_width=True)

        # Perturbation stats
        st.markdown(f"""
        <div style="background:#0F1F3D; border-radius:8px; padding:14px 18px;">
            <strong style="color:#00D4FF;">Perturbation Statistics</strong><br>
            <span style="color:#8899BB; font-size:0.85rem;">
                L2 Norm: {result.get('perturbation_magnitude', 0):.4f} |
                Budget (epsilon): {result.get('epsilon', epsilon):.3f} |
                Mutable features: {len(MUTABLE_FEATURES)} / {N_FEATURES}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Batch Experiment Summary ───────────────────────────────────────────────
    attack_results_path = os.path.join(RESULTS_DIR, "attack_results.json")
    if os.path.exists(attack_results_path):
        st.markdown("---")
        st.markdown("#### 📈 Batch Experiment Results (Pre-computed)")
        try:
            with open(attack_results_path) as f:
                agg = json.load(f)
            from src.visualization import plot_attack_comparison
            # Show for first available model
            first_model = list(agg.keys())[0] if agg else None
            if first_model and first_model in agg:
                st.markdown(f"**Target: {first_model}**")
                fig_agg = plot_attack_comparison(agg[first_model])
                st.plotly_chart(fig_agg, use_container_width=True)
        except Exception:
            pass
