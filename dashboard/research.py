"""
research.py — Dashboard Page 6: Research & Documentation
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    st.markdown("""
    <div style="background:#0F1F3D; border-left:4px solid #00E676;
                border-radius:8px; padding:16px 20px; margin-bottom:20px;">
        <h2 style="color:#00E676; margin:0; font-size:1.5rem;">📚 Research & Documentation</h2>
        <p style="color:#8899BB; margin:6px 0 0 0; font-size:0.9rem;">
            Full academic documentation — abstract, methodology, algorithms, results, and references.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab navigation ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Abstract", "Problem Statement", "Methodology",
        "Attack Algorithms", "Defense Techniques", "Limitations & Future", "References"
    ])

    # ── Tab 1: Abstract ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        ## Abstract

        Machine learning (ML) has emerged as a powerful paradigm for automated malware detection,
        offering improved scalability and detection rates over traditional signature-based approaches.
        However, ML-based detectors are inherently vulnerable to **adversarial examples** — carefully
        crafted inputs designed to fool classifiers while remaining functionally equivalent to their
        original form.

        This work presents **MalwareShield AI**, an interactive academic platform for studying the
        adversarial robustness of ML-based malware detection systems. The platform implements:

        1. Three baseline classifiers (Random Forest, Gradient Boosting, Logistic Regression) trained
           on a synthetic PE-feature dataset with realistic statistical distributions.
        2. Four adversarial attack simulations (FGSM, PGD, Genetic Algorithm, Random perturbation)
           operating exclusively in feature space.
        3. Three defense mechanisms (Adversarial Training, Input Validation, Ensemble Defense).
        4. A comprehensive robustness evaluation framework computing clean accuracy, adversarial
           accuracy, attack success rates, and standard classification metrics.
        5. An interactive Streamlit dashboard for real-time experimentation and visualization.

        > **Research Disclaimer:** All experiments use a synthetic demonstration dataset.
        > No real malware was downloaded, executed, or distributed. Feature-space perturbations
        > do not constitute functional PE modifications.
        """)

    # ── Tab 2: Problem Statement ───────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        ## Problem Statement

        ### The Challenge of Adversarial Malware Detection

        As ML-based antivirus solutions become widespread, adversaries increasingly develop
        **adversarial evasion techniques** targeting the ML classification layer. The core challenge is:

        > *Given a trained malware classifier f(x) → {Benign, Malicious}, can an attacker produce
        > a modified feature vector x' such that f(x') = Benign while x' still represents malicious behavior?*

        ### Why This Matters

        - **Gradient-based attacks** (FGSM, PGD) can fool differentiable classifiers in a single forward/backward pass.
        - **Transfer attacks** allow adversarial examples crafted against a surrogate model to transfer to black-box models.
        - **Genetic/evolutionary attacks** require no gradient information, targeting non-differentiable tree ensembles.
        - Standard model training ignores adversarial examples, leaving deployed systems vulnerable.

        ### Research Gap

        While adversarial attacks are well-studied in image classification, the **malware detection domain**
        presents unique constraints:
        - Feature-space vs. problem-space distinction (not all feature perturbations map to valid executables)
        - Non-differentiable nature of tree-based classifiers (industry standard)
        - Class imbalance and label noise in real datasets
        - Practical computational budgets for deploying defenses at scale
        """)

        # Threat model diagram
        st.markdown("### Threat Model")
        fig = go.Figure()
        threat_nodes = [
            ("Attacker\nGoal", 0.1, 0.5, "#FF1744", 20),
            ("Feature Space\nManipulation", 0.3, 0.7, "#FF6B35", 16),
            ("Problem Space\nConstraint", 0.3, 0.3, "#FFD600", 16),
            ("Surrogate\nModel", 0.5, 0.8, "#7CB9E8", 14),
            ("Target\nClassifier", 0.7, 0.5, "#00D4FF", 18),
            ("Evasion\nResult", 0.9, 0.5, "#00E676", 16),
        ]
        edges = [(0,1),(0,2),(1,3),(1,4),(3,4),(4,5)]
        xs = [n[1] for n in threat_nodes]
        ys = [n[2] for n in threat_nodes]
        for s, d in edges:
            fig.add_trace(go.Scatter(x=[xs[s],xs[d]], y=[ys[s],ys[d]],
                                      mode="lines", line=dict(color="rgba(0,212,255,0.3)", width=2),
                                      showlegend=False, hoverinfo="none"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=[n[4] for n in threat_nodes],
                        color=[n[3] for n in threat_nodes], opacity=0.9),
            text=[n[0] for n in threat_nodes],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            showlegend=False,
        ))
        fig.update_layout(height=280, paper_bgcolor="#0A1628", plot_bgcolor="#0A1628",
                           xaxis=dict(visible=False, range=[-0.05, 1.05]),
                           yaxis=dict(visible=False, range=[0.1, 0.95]),
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Methodology ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
        ## Methodology

        ### 1. Dataset Generation

        A synthetic PE-feature dataset was generated with **10,000 samples** and **50 features**
        modeled after real PE file analysis outputs. Feature groups:

        | Feature Group | Indices | Description |
        |---|---|---|
        | PE Header | 0–9 | num_sections, image_size, entry_RVA, alignment, debug flag |
        | Section Stats | 10–19 | Per-section entropy values |
        | Import/Export | 20–29 | DLL counts, function counts, import entropy |
        | File Entropy | 30–37 | Overall file entropy distribution |
        | String Features | 38–43 | URL count, registry refs, crypto API, packing indicators |
        | Misc | 44–49 | Resource count, overlay, TLS, signature score, API depth |

        Benign samples use moderate entropy (4–7), fewer suspicious strings, and normal DLL counts.
        Malicious samples use high entropy (6–8, simulating packing), more suspicious strings, fewer imports.

        ### 2. Preprocessing Pipeline

        ```
        Raw Features → StandardScaler (fit on train) → Train/Val/Test Split (70/15/15)
        ```
        - Stratified split preserves class ratios
        - Scaler fitted exclusively on training data to prevent leakage
        - Fixed random seed (42) for full reproducibility

        ### 3. Mutable vs Immutable Features

        | Category | Features | Mutable? |
        |---|---|---|
        | PE Header & Section Structure | 0–19 | ❌ No (structural integrity) |
        | Import/Export, Entropy, Strings, Misc | 20–49 | ✅ Yes (can vary without breaking PE structure) |

        This constraint models the assumption that attackers can alter observable behavioral signals
        while maintaining a valid (or apparently valid) file structure.

        ### 4. Model Selection

        Best model selection uses **validation F1-score** (not test set) to prevent bias.
        """)

    # ── Tab 4: Attack Algorithms ──────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
        ## Attack Algorithms

        ### A. FGSM — Fast Gradient Sign Method

        **Origin:** Goodfellow et al., 2014 (ICLR 2015).

        **Principle:** Compute gradient of loss w.r.t. input, perturb in gradient sign direction.

        ```python
        x_adv = x + ε · sign(∇_x L(f(x), y))
        ```

        **Implementation Note:** Since tree models (Random Forest, Gradient Boosting) are
        non-differentiable, we use **Logistic Regression as a differentiable surrogate**.
        The gradient `coef_[0]` from LR approximates the perturbation direction.
        This is a **transfer attack** — adversarial examples crafted against LR are tested
        on the target model.

        ---

        ### B. PGD — Projected Gradient Descent

        **Origin:** Madry et al., 2018 (ICLR).

        **Principle:** Iterative FGSM with projection back to epsilon-ball after each step.

        ```python
        x^(0) = x + U(-ε, ε)   # random start
        for t in range(T):
            x^(t+1) = Proj(x^(t) + α · sign(∇_x L))
        ```

        Parameters: ε=0.1 (budget), α=0.02 (step size), T=40 (iterations).

        ---

        ### C. Genetic Algorithm Attack

        **Principle:** Black-box evolutionary search. No gradient required.

        ```
        Population = random perturbations in ε-ball
        Fitness    = P(malicious | x_adv)       [maximize]
        Selection  = top-k by fitness
        Crossover  = single-point on mutable indices
        Mutation   = random delta on mutable features with rate p_m
        ```

        Parameters: Pop=50, Gen=30, Mutation rate=0.1.

        **Advantage:** Works against non-differentiable tree ensembles without surrogate.

        ---

        ### D. Random Perturbation (Control Baseline)

        ```python
        x_adv = x + U(-ε, ε) on mutable features only
        ```

        Establishes a lower-bound for attack success rate. Any directed attack should
        outperform this baseline; if it does not, the directed attack is not effective.
        """)

    # ── Tab 5: Defense Techniques ─────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
        ## Defense Techniques

        ### 1. Adversarial Training

        **Principle:** Augment the training set with adversarial examples so the model
        learns to correctly classify both clean and adversarially perturbed samples.

        ```
        1. Train preliminary Random Forest on clean data
        2. Apply PGD to 30% of training samples (mutable features only)
        3. Retrain Random Forest on X_clean ∪ X_adversarial (keeping original labels)
        ```

        **Trade-off:** May slightly reduce clean accuracy; significantly improves robustness.

        ---

        ### 2. Input Validation / Feature Sanitization

        **Principle:** Clip features to bounds observed in clean training data before prediction.

        ```
        lower_i = mean_i - 3·σ_i
        upper_i = mean_i + 3·σ_i
        x_sanitized = clip(x, lower, upper)
        ```

        **Trade-off:** Negligible clean accuracy impact. Effective against large-magnitude attacks.
        Less effective against attacks within normal feature bounds.

        ---

        ### 3. Ensemble-Based Defense

        **Principle:** Soft-voting ensemble of diverse classifiers. Adversarial examples must
        fool Random Forest, Gradient Boosting, AND Logistic Regression simultaneously.

        ```
        P(malicious) = (P_RF + P_GB + P_LR) / 3
        ```

        **Trade-off:** Higher computation cost (3× inference). Diversity of base models makes
        simultaneous evasion significantly harder.

        ---

        ### Comparison Summary

        | Defense | Clean Acc Impact | Robustness Gain | Computational Cost |
        |---|---|---|---|
        | Adversarial Training | Low (1–3%) | High | High (training) |
        | Input Validation | Minimal | Medium | Low |
        | Ensemble | Minimal | Medium-High | High (inference) |
        """)

    # ── Tab 6: Limitations & Future ───────────────────────────────────────────
    with tabs[5]:
        st.markdown("""
        ## Limitations

        ### Current Limitations

        1. **Synthetic Dataset:** Results are on a synthetic dataset, not real malware.
           Real-world feature distributions are more complex and noisy. Performance
           numbers (99–100% accuracy) reflect the artificial separability of the synthetic data.

        2. **Feature-Space Only:** Attacks operate in feature space, not problem space.
           Generating actual adversarial PE files that maintain functionality while evading
           detection requires additional constraints (PE structure validity, code equivalence).

        3. **Gradient Transfer Assumption:** FGSM/PGD use gradient transfer from LR to tree models.
           Transfer success rates vary in practice depending on model diversity and feature space geometry.

        4. **No Obfuscation Attacks:** Real-world malware evasion includes code obfuscation,
           packing, and API call reordering — not modeled here.

        5. **Static Analysis Only:** Dynamic behavioral features (API call sequences, network traffic,
           memory access patterns) are not included.

        ---

        ## Future Enhancements

        1. **EMBER Dataset Integration:** Use the real EMBER dataset for production-quality experiments.
        2. **Problem-Space Attacks:** Implement GAMMA or MAZE attacks that modify actual PE files
           while preserving functionality.
        3. **Deep Learning Models:** Implement MalConv (raw byte CNN) or Transformer-based detectors.
        4. **Certified Robustness:** Apply randomized smoothing to provide robustness certificates.
        5. **Adaptive Attacks:** Implement attacks that are aware of the defense mechanism.
        6. **Dynamic Analysis:** Integrate sandbox execution traces as behavioral features.
        7. **Online Learning:** Continuous model retraining as new malware families emerge.
        8. **Explainability:** SHAP values for model interpretability in SOC workflows.
        """)

    # ── Tab 7: References ─────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("""
        ## References

        1. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2014).
           *Explaining and Harnessing Adversarial Examples.*
           ICLR 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)

        2. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.** (2017).
           *Towards Deep Learning Models Resistant to Adversarial Attacks.*
           ICLR 2018. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

        3. **Anderson, H. S., & Roth, P.** (2018).
           *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models.*
           arXiv:1804.04637. [GitHub](https://github.com/elastic/ember)

        4. **Grosse, K., Papernot, N., Manoharan, P., Backes, M., & McDaniel, P.** (2017).
           *Adversarial Examples for Malware Detection.*
           ESORICS 2017.

        5. **Suciu, O., Coull, S. E., & Johns, J.** (2019).
           *Exploring Adversarial Examples in Malware Detection.*
           IEEE S&P Deep Learning Security Workshop.

        6. **Demetrio, L., Coull, S., Biggio, B., Lagorio, G., Armando, A., & Roli, F.** (2021).
           *Functionality-Preserving Black-Box Optimization of Adversarial Windows Malware.*
           IEEE TIFS.

        7. **Pierazzi, F., Pendlebury, F., Cortellazzi, J., & Cavallaro, L.** (2020).
           *Intriguing Properties of Adversarial ML Attacks in the Problem Space.*
           IEEE S&P 2020.

        8. **Nicolae, M-I., et al.** (2018).
           *Adversarial Robustness Toolbox v1.0.0.*
           arXiv:1807.01069. [GitHub](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

        9. **Scikit-learn: Machine Learning in Python.** Pedregosa et al., JMLR 12, 2825-2830, 2011.

        10. **Chen, T., & Guestrin, C.** (2016).
            *XGBoost: A Scalable Tree Boosting System.*
            KDD 2016.
        """)
