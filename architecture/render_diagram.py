"""
Generates high-resolution architecture diagram for architecture/system-overview.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Title & Subtitle
    ax.text(
        0.5, 0.95,
        "Neuro-Symbolic AI Framework for Scientific Model Selection",
        color="#f0f6fc", fontsize=20, weight="bold", ha="center", va="center"
    )
    ax.text(
        0.5, 0.91,
        "Hybrid Architecture: Mechanistic ODEs + Temporal Attention GRU + Structured LLM Reasoning + Deterministic Guardrails",
        color="#8b949e", fontsize=11, ha="center", va="center"
    )

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, text_color="#f0f6fc", items=None):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=bg_color, edgecolor=border_color, linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.04, title, color=text_color, fontsize=12, weight="bold", ha="center", va="top")
        if subtitle:
            ax.text(x + w/2, y + h - 0.08, subtitle, color="#8b949e", fontsize=8.5, style="italic", ha="center", va="top")
        
        if items:
            for i, item in enumerate(items):
                ax.text(x + 0.02, y + h - 0.13 - i * 0.038, f"• {item}", color="#c9d1d9", fontsize=8.5, va="top")

    # Helper for drawing flow arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2.2, mutation_scale=18)
        )
        if label:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mid_x, mid_y + 0.018, label, color="#79c0ff", fontsize=8, ha="center", weight="bold")

    # 1. INPUT DATA LAYER (Left Top)
    draw_box(
        0.04, 0.58, 0.25, 0.28,
        "1. Experimental Trajectory",
        "Raw Observations & Kinetics",
        "#161b22", "#30363d", "#58a6ff",
        [
            "Time-series (OD600 / biomass)",
            "Sparse / noisy sampling points",
            "Finite differences: dy/dt, d²y/dt²",
            "Zero-leakage data isolation"
        ]
    )

    # 2. MECHANISTIC ODE REGRESSION (Center-Left Top)
    draw_box(
        0.36, 0.62, 0.28, 0.24,
        "2A. Mechanistic ODE Suite",
        "Deterministic Non-linear Least Squares",
        "#161b22", "#238636", "#3fb950",
        [
            "Modified Gompertz & Logistic",
            "Richards (asymmetry ν factor)",
            "Baranyi-Roberts dynamic model",
            "Loss: AIC, AICc, BIC, RSS, R²"
        ]
    )

    # 3. TEMPORAL NEURAL ENCODER (Center-Left Bottom)
    draw_box(
        0.36, 0.33, 0.28, 0.25,
        "2B. Temporal Neural Encoder",
        "PyTorch BiGRU + Attention Pooling",
        "#161b22", "#1f6feb", "#58a6ff",
        [
            "Bidirectional Recurrent GRU",
            "Multi-Head Temporal Attention",
            "Dynamic inflection detection",
            "Latent kinetic representation"
        ]
    )

    # 4. STRUCTURED LLM REASONING & CRITIQUE (Center-Right)
    draw_box(
        0.70, 0.44, 0.26, 0.42,
        "3. Symbolic Reasoning Engine",
        "Structured Hypotheses & Multi-Agent Critique",
        "#161b22", "#a371f7", "#d2a8ff",
        [
            "Strict JSON schema enforcement",
            "Parsimony & penalty reasoning",
            "Domain microbiological priors",
            "Adversarial critique agent",
            "Arbitration & fallback solver"
        ]
    )

    # 5. DETERMINISTIC GUARDRAILS (Bottom Center)
    draw_box(
        0.36, 0.05, 0.28, 0.23,
        "4. Deterministic Guardrails",
        "Safety Boundaries & Hallucination Filter",
        "#161b22", "#da3633", "#f85149",
        [
            "Biophysical parameter bounds (μ > 0, λ ≥ 0)",
            "Asymptotic monotonicity check",
            "Parameter hallucination detector",
            "Automated fallback / override"
        ]
    )

    # 6. AUDIT & PROVENANCE (Bottom Right)
    draw_box(
        0.70, 0.05, 0.26, 0.23,
        "5. Cryptographic Provenance",
        "Reproducibility & Governance Layer",
        "#161b22", "#d29922", "#e3b341",
        [
            "SHA-256 input / prompt digest",
            "Deterministic run ledger (.jsonl)",
            "Execution latency tracking",
            "Responsible asset governance"
        ]
    )

    # ARROWS & CONNECTIONS
    # Input -> ODE Suite & Temporal Encoder
    draw_arrow(0.29, 0.74, 0.36, 0.74, "Data Points")
    draw_arrow(0.29, 0.65, 0.36, 0.45, "Feature Vector")

    # ODE Suite -> LLM Engine
    draw_arrow(0.64, 0.74, 0.70, 0.74, "AIC / R² / Params")

    # Neural Encoder -> LLM Engine
    draw_arrow(0.64, 0.45, 0.70, 0.55, "Temporal Evidence")

    # LLM Engine -> Guardrails
    draw_arrow(0.75, 0.44, 0.55, 0.28, "Proposed Hypothesis")

    # ODE Suite -> Guardrails (Ground truth params)
    draw_arrow(0.50, 0.62, 0.50, 0.28, "Fitted Ground Truth")

    # Guardrails -> Provenance
    draw_arrow(0.64, 0.16, 0.70, 0.16, "Audit Verdict")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    output_path = Path("architecture/system-overview.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Success] Saved architecture diagram to {output_path.resolve()}")

if __name__ == "__main__":
    generate_architecture_diagram()
