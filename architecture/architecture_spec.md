# System Architecture Specification

## Overview

The **Neuro-Symbolic AI Framework for Scientific Model Selection** couples mechanistic differential equations with deep recurrent neural representations and structured large language model (LLM) reasoning, wrapped inside formal deterministic guardrails and cryptographic provenance tracking.

```mermaid
flowchart TD
    subgraph Data["1. Trajectory Data Ingestion"]
        RAW["Microbial Growth Trajectory (t, OD600)"] --> DIFF["Finite Differences (dy/dt, d²y/dt²)"]
    end

    subgraph Mechanistic["2A. Mechanistic ODE Suite"]
        DIFF --> GOMP["Modified Gompertz Fit"]
        DIFF --> LOGI["Logistic Model Fit"]
        DIFF --> RICH["Richards Model Fit (ν shape)"]
        DIFF --> BARA["Baranyi-Roberts Model Fit"]
        GOMP & LOGI & RICH & BARA --> STATS["Loss / AIC / BIC / R² Engine"]
    end

    subgraph Neural["2B. Deep Temporal Evidence Encoder"]
        DIFF --> GRU["Bidirectional PyTorch GRU"]
        GRU --> ATTN["Multi-Head Temporal Attention Pooling"]
        ATTN --> PRIOR["Neural Kinetic Priors & Latent Embeddings"]
    end

    subgraph Reasoning["3. Symbolic Hypothesis & Critique"]
        STATS & PRIOR --> PROMPT["Structured Scientific Prompt Builder"]
        PROMPT --> HYP["Hypothesis Generation Agent (JSON Schema)"]
        HYP --> CRIT["Adversarial Critique Agent"]
        CRIT --> ARB["Arbitration & Complexity Solver"]
    end

    subgraph Guardrails["4. Deterministic Guardrail Layer"]
        ARB --> BOUNDS["Biophysical Parameter Bounds (μ > 0, λ ≥ 0)"]
        ARB --> MONO["Asymptotic Monotonicity Validator"]
        STATS --> HALLUC["Parameter Hallucination Filter"]
        BOUNDS & MONO & HALLUC --> VERDICT{"Pass All Rules?"}
        VERDICT -- "No" --> CORR["Fallback to Parsimonious AIC Ground Truth"]
        VERDICT -- "Yes" --> PASS["Approved Scientific Decision"]
    end

    subgraph Provenance["5. Cryptographic Provenance Layer"]
        CORR & PASS --> HASH["SHA-256 Digest Generator"]
        HASH --> AUDIT["Immutable Audit Ledger (JSON Lines)"]
    end

    style Data fill:#161b22,stroke:#30363d,color:#58a6ff
    style Mechanistic fill:#161b22,stroke:#238636,color:#3fb950
    style Neural fill:#161b22,stroke:#1f6feb,color:#58a6ff
    style Reasoning fill:#161b22,stroke:#a371f7,color:#d2a8ff
    style Guardrails fill:#161b22,stroke:#da3633,color:#f85149
    style Provenance fill:#161b22,stroke:#d29922,color:#e3b341
```

---

## Subsystem Specifications

### 1. Mechanistic ODE Non-linear Regression Engine
- **Supported Formalisms:**
  - **Modified Gompertz:** Asymmetric sigmoidal curve capturing rapid acceleration and exponential deceleration.
  - **Logistic (Verhulst):** Symmetric sigmoidal curve with inflection at mid-point.
  - **Richards:** Generalized 5-parameter curve introducing shape factor $\nu$ for arbitrary inflection points.
  - **Baranyi-Roberts:** Mechanistic kinetic formulation incorporating dynamic physiological adjustment function $A(t)$.
- **Optimization Algorithms:** Bounded Trust Region Reflective (`scipy.optimize.curve_fit(method='trf')`) with empirical parameter initialization from derivative peaks.
- **Statistical Criteria:** Residual Sum of Squares (RSS), $R^2$, Akaike Information Criterion (AIC), Hurvich-Tsai small-sample corrected criterion (AICc), Bayesian Information Criterion (BIC), and Akaike weights ($w_i$).

### 2. Deep Temporal Evidence Encoder (GRU + Attention)
- **Model Topology:** 2-layer Bidirectional Recurrent GRU (PyTorch) with LayerNorm and GELU activations.
- **Input Channels (4D):** $[y(t), t / t_{max}, \frac{dy}{dt}, \frac{d^2y}{dt^2}]$.
- **Multi-Head Self-Attention Pooling:** 4 parallel attention heads projecting temporal hidden vectors to identify critical kinetic transitions (lag exit, maximum specific growth velocity, deceleration).
- **Output Embeddings:** Latent kinetic representation vector ($D=32$) and softmax prior probability distribution over candidate model families.

### 3. Structured LLM Scientific Reasoning Engine
- **Schema Enforcement:** Pydantic models enforcing strict JSON outputs (`ModelHypothesis`, `CritiqueReport`, `ArbitrationDecision`).
- **Prompt Architecture:** Multi-modal prompt syntax binding deterministic numerical fits, information criteria, parameter standard errors, and neural attention transition timings.
- **Multi-Agent Workflow:**
  1. *Primary Reasoner:* Formulates mechanistic hypothesis and biological interpretation.
  2. *Adversarial Critique Agent:* Scrutinizes model complexity over-fitting (e.g., penalizing Richards shape parameter $\nu$ if $\Delta AIC < 2.0$).
  3. *Arbitrator:* Resolves discrepancies between heuristic intuition and empirical evidence.

### 4. Deterministic Guardrails & Hallucination Filter
- **Biophysical Validation:** Hard boundaries on maximum specific growth rate ($\mu_{max} \in [0.001, 3.5]\text{ h}^{-1}$), lag time ($\lambda \in [0, 36]\text{ h}$), and initial optical density ($y_0 \ge 0$).
- **Hallucination Detection:** Compares numerical parameter citations in the LLM output with deterministic solver outputs; flags any deviation exceeding $\pm 5\%$.
- **Automated Fallback:** If any critical rule fails, execution immediately rolls back to the parsimonious minimum-AIC ground-truth model.

### 5. Cryptographic Provenance & Audit Trail
- **Hashing Algorithm:** Canonical JSON serialization hashed via SHA-256.
- **Tracked Entities:** Input trajectory vector, solver parameter dictionary, neural embedding digest, prompt tokens, LLM response payload, and guardrail audit report.
- **Audit Persistence:** Append-only JSONL log guaranteeing byte-level reproducibility for scientific reviews and regulatory compliance.
