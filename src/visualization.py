"""
visualization.py — Chart generation utilities (Plotly + Matplotlib)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#7B2FBE",
    "success": "#00E676",
    "danger": "#FF1744",
    "warning": "#FFD600",
    "bg": "#0A1628",
    "card": "#0F1F3D",
    "text": "#E8EAF6",
    "grid": "#1A2744",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)


def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, model_name: str = "") -> go.Figure:
    """Interactive Plotly confusion matrix heatmap."""
    cm_arr = np.array(cm)
    labels = ["Benign", "Malicious"]
    text = [[f"TN: {cm_arr[0,0]}", f"FP: {cm_arr[0,1]}"],
            [f"FN: {cm_arr[1,0]}", f"TP: {cm_arr[1,1]}"]]

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale=[[0, COLORS["card"]], [1, COLORS["primary"]]],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14, color=COLORS["text"]),
        showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    return apply_template(fig)


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curves(roc_data_dict: dict) -> go.Figure:
    """Plot multiple ROC curves on one chart."""
    fig = go.Figure()
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Random Classifier",
        showlegend=True,
    ))
    colors_list = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["secondary"]]
    for i, (name, data) in enumerate(roc_data_dict.items()):
        if not data:
            continue
        color = colors_list[i % len(colors_list)]
        auc = data.get("auc", 0)
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return apply_template(fig)


# ── Feature Importance ─────────────────────────────────────────────────────────

def plot_feature_importance(df_importance: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    df_top = df_importance.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df_top["importance"],
        y=df_top["feature"],
        orientation="h",
        marker=dict(
            color=df_top["importance"],
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(400, top_n * 22),
    )
    return apply_template(fig)


# ── Attack Success Rate Comparison ────────────────────────────────────────────

def plot_attack_comparison(attack_results: dict, model_names: list = None) -> go.Figure:
    """Bar chart comparing attack success rates across methods."""
    methods = list(attack_results.keys())
    asr = [attack_results[m].get("attack_success_rate", 0) * 100 for m in methods]

    fig = go.Figure(go.Bar(
        x=methods,
        y=asr,
        marker=dict(
            color=asr,
            colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
            showscale=True,
            colorbar=dict(title="Success %"),
        ),
        text=[f"{v:.1f}%" for v in asr],
        textposition="outside",
    ))
    fig.update_layout(
        title="Attack Success Rates by Method",
        xaxis_title="Attack Method",
        yaxis_title="Attack Success Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    return apply_template(fig)


# ── Accuracy Comparison ────────────────────────────────────────────────────────

def plot_accuracy_comparison(metrics_dict: dict) -> go.Figure:
    """Grouped bar chart: clean vs adversarial accuracy by model."""
    models = list(metrics_dict.keys())
    clean_acc = [metrics_dict[m].get("clean_accuracy", 0) * 100 for m in models]
    adv_acc = [metrics_dict[m].get("adversarial_accuracy", 0) * 100 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Clean Accuracy",
        x=models, y=clean_acc,
        marker_color=COLORS["primary"],
        text=[f"{v:.1f}%" for v in clean_acc],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Adversarial Accuracy",
        x=models, y=adv_acc,
        marker_color=COLORS["danger"],
        text=[f"{v:.1f}%" for v in adv_acc],
        textposition="outside",
    ))
    fig.update_layout(
        title="Clean vs Adversarial Accuracy",
        barmode="group",
        yaxis=dict(range=[0, 115]),
        yaxis_title="Accuracy (%)",
    )
    return apply_template(fig)


# ── Perturbation Strength vs Accuracy ─────────────────────────────────────────

def plot_robustness_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    surrogate_model,
    epsilons: list = None,
) -> go.Figure:
    """Plot adversarial accuracy vs epsilon (perturbation strength)."""
    from src.attacks import pgd_attack, run_attack_experiment

    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

    from sklearn.metrics import accuracy_score
    clean_acc = accuracy_score(y_test, model.predict(X_test))

    adv_accs = []
    for eps in epsilons:
        try:
            exp = run_attack_experiment(
                pgd_attack, surrogate_model, model, X_test, y_test,
                n_samples=100, seed=42, epsilon=eps
            )
            asr = exp.get("attack_success_rate", 0)
            adv_accs.append(max(0.0, clean_acc * (1 - asr)))
        except Exception:
            adv_accs.append(clean_acc)

    fig = go.Figure()
    fig.add_hline(y=clean_acc, line_dash="dash", line_color=COLORS["success"],
                  annotation_text=f"Clean Acc: {clean_acc:.3f}")
    fig.add_trace(go.Scatter(
        x=epsilons, y=adv_accs,
        mode="lines+markers",
        name="Adversarial Accuracy (PGD)",
        line=dict(color=COLORS["danger"], width=2),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title="Robustness: Adversarial Accuracy vs Perturbation Strength (ε)",
        xaxis_title="Epsilon (ε) — Perturbation Magnitude",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05]),
    )
    return apply_template(fig)


# ── Model Metrics Summary ─────────────────────────────────────────────────────

def plot_metrics_radar(metrics_dict: dict) -> go.Figure:
    """Radar chart comparing models across multiple metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors_list = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["secondary"]]

    fig = go.Figure()
    for i, (name, m) in enumerate(metrics_dict.items()):
        values = [m.get(k, 0) or 0 for k in metric_keys]
        values_closed = values + [values[0]]
        labels_closed = metric_labels + [metric_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name=name,
            line=dict(color=colors_list[i % len(colors_list)]),
            opacity=0.6,
        ))
    fig.update_layout(
        title="Model Performance Radar Chart",
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["grid"]),
            angularaxis=dict(gridcolor=COLORS["grid"]),
        ),
    )
    return apply_template(fig)


# ── Confidence Distribution ───────────────────────────────────────────────────

def plot_confidence_shift(original_confs: list, adversarial_confs: list,
                           attack_name: str = "") -> go.Figure:
    """Histogram comparing prediction confidence before vs after attack."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=original_confs, name="Original", nbinsx=20,
        marker_color=COLORS["primary"], opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=adversarial_confs, name="Adversarial", nbinsx=20,
        marker_color=COLORS["danger"], opacity=0.7,
    ))
    fig.update_layout(
        title=f"Prediction Confidence Distribution — {attack_name}",
        xaxis_title="Confidence",
        yaxis_title="Count",
        barmode="overlay",
    )
    return apply_template(fig)


# ── Feature Perturbation Visualizer ──────────────────────────────────────────

def plot_feature_perturbation(x_orig: np.ndarray, x_adv: np.ndarray,
                               feature_names: list, top_n: int = 15) -> go.Figure:
    """Show delta (perturbation) per feature."""
    delta = x_adv - x_orig
    df = pd.DataFrame({"feature": feature_names, "delta": delta})
    df["abs_delta"] = df["delta"].abs()
    df_top = df.nlargest(top_n, "abs_delta").sort_values("delta")

    colors = [COLORS["danger"] if d > 0 else COLORS["primary"] for d in df_top["delta"]]
    fig = go.Figure(go.Bar(
        x=df_top["delta"],
        y=df_top["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in df_top["delta"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=1)
    fig.update_layout(
        title=f"Top {top_n} Feature Perturbations",
        xaxis_title="Perturbation (Δ)",
        yaxis_title="Feature",
        height=max(400, top_n * 25),
    )
    return apply_template(fig)
