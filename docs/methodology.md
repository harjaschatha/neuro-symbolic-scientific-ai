# Mathematical Formulation & Scientific Methodology

## 1. Mechanistic ODE Growth Formulations

Microbial biomass growth in batch culture is traditionally modeled via non-linear ordinary differential equations (ODEs). In this framework, we standardize parameterization based on Zwietering et al. (1990) and Baranyi & Roberts (1994) where:
- $y(t)$: Optical density ($\text{OD}_{600}$) or logarithm of relative population size $\ln(N(t)/N_0)$ at time $t$
- $y_0$: Initial baseline optical density / inoculum size
- $A$: Asymptotic maximum population increase ($\lim_{t \to \infty} y(t) - y_0$)
- $\mu_{max}$: Maximum specific growth rate ($\text{h}^{-1}$), defined as $\max_{t} \frac{1}{y} \frac{dy}{dt}$
- $\lambda$: Lag time duration ($\text{h}$), defined as the x-intercept of the tangent line through the inflection point

---

### 1.1 Modified Gompertz Model
The modified Gompertz equation describes asymmetric sigmoidal kinetics with rapid deceleration following the inflection point:

$$y(t) = y_0 + A \exp\left\{ -\exp\left[ \frac{\mu_{max} e}{A} (\lambda - t) + 1 \right] \right\}$$

where $e = \exp(1) \approx 2.71828$. The inflection point occurs at:

$$t_{inf} = \lambda + \frac{A}{\mu_{max} e}$$

$$\left. y(t) \right|_{t = t_{inf}} = y_0 + \frac{A}{e}$$

---

### 1.2 Modified Logistic Model
The modified Logistic equation (Verhulst model) represents symmetric sigmoidal growth where maximum velocity occurs precisely at half the carrying capacity:

$$y(t) = y_0 + \frac{A}{1 + \exp\left[ \frac{4 \mu_{max}}{A} (\lambda - t) + 2 \right]}$$

The inflection point occurs at:

$$t_{inf} = \lambda + \frac{A}{2 \mu_{max}}$$

---

### 1.3 Modified Richards Model
The Richards equation introduces an empirical shape parameter $\nu > 0$ that modulates the position of the inflection point and curve asymmetry:

$$y(t) = y_0 + A \left\{ 1 + \nu \exp(1 + \nu) \exp\left[ \frac{\mu_{max}}{A} (1 + \nu)^{1 + 1/\nu} (\lambda - t) \right] \right\}^{-1/\nu}$$

When $\nu \to 0$, the Richards curve asymptotically approaches the Gompertz model; when $\nu = 1$, it reduces to the Logistic curve.

---

### 1.4 Baranyi and Roberts Dynamic Model
The Baranyi-Roberts model introduces a physiological state adjustment variable $\alpha(t)$ representing critical intracellular substance synthesis during the lag phase:

$$\frac{dy}{dt} = \mu_{max} \cdot \alpha(t) \cdot \left[ 1 - \exp(y - y_{max}) \right]$$

$$\frac{d\alpha}{dt} = \mu_{max} \alpha(t) [1 - \alpha(t)]$$

The closed-form analytical solution is:

$$y(t) = y_0 + \mu_{max} A(t) - \ln\left[ 1 + \frac{\exp(\mu_{max} A(t)) - 1}{\exp(A)} \right]$$

$$A(t) = t + \frac{1}{\mu_{max}} \ln\left[ \exp(-\mu_{max} t) + \exp(-\mu_{max} \lambda) - \exp(-\mu_{max}(t + \lambda)) \right]$$

---

## 2. Statistical Model Selection Criteria

For an empirical time series with $N$ observations $\mathbf{y} = [y_1, \dots, y_N]^T$ and $K$ estimated parameters $\hat{\boldsymbol{\theta}}$, we compute:

### Residual Sum of Squares (RSS):
$$\text{RSS} = \sum_{i=1}^N \left( y_i - \hat{y}(t_i; \hat{\boldsymbol{\theta}}) \right)^2$$

### Coefficient of Determination ($R^2$):
$$R^2 = 1 - \frac{\text{RSS}}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

### Log-Likelihood Function (Gaussian residuals):
$$\ln \mathcal{L}(\hat{\boldsymbol{\theta}}) = -\frac{N}{2} \left[ \ln(2\pi) + \ln\left(\frac{\text{RSS}}{N}\right) + 1 \right]$$

### Information Criteria:
- **Akaike Information Criterion (AIC):**
  $$\text{AIC} = 2K - 2\ln \mathcal{L} = N \ln\left(\frac{\text{RSS}}{N}\right) + 2K + N(1 + \ln(2\pi))$$

- **Corrected AIC for Small Samples (AICc):**
  $$\text{AICc} = \text{AIC} + \frac{2K(K+1)}{N - K - 1}$$

- **Bayesian Information Criterion (BIC):**
  $$\text{BIC} = K \ln(N) - 2\ln \mathcal{L}$$

- **Akaike Weights ($w_i$):**
  $$\Delta_i = \text{AICc}_i - \min_j(\text{AICc}_j)$$
  $$w_i = \frac{\exp\left(-\frac{1}{2}\Delta_i\right)}{\sum_{m=1}^M \exp\left(-\frac{1}{2}\Delta_m\right)}$$

---

## 3. Deep Temporal Attention Representation

Let the sequence of empirical measurements be represented as:
$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times 4}$$
where $\mathbf{x}_t = \left[ y_t, \frac{t}{t_{max}}, \frac{\Delta y_t}{\Delta t}, \frac{\Delta^2 y_t}{\Delta t^2} \right]$.

The sequence is processed by a 2-layer Bidirectional GRU:
$$\mathbf{h}_t = \text{BiGRU}(\mathbf{x}_t, \mathbf{h}_{t-1})$$

A multi-head self-attention pooling operator computes temporal dynamic weights $\alpha_t$:
$$\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K$$
$$\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}$$
$$\boldsymbol{\alpha} = \text{softmax}(\mathbf{S})$$
$$\mathbf{c} = \sum_{t=1}^T \bar{\alpha}_t \mathbf{h}_t$$

The context vector $\mathbf{c}$ parameterizes the model classification head $P(M_k | \mathbf{X}) = \text{softmax}(\mathbf{W}_c \mathbf{c} + \mathbf{b}_c)$.

---

## 4. Multi-Agent Reasoning, Critique & Arbitration Protocol

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Hypothesis Generation Agent                              │
│    Input: ODE fits + GRU Priors + Transition Attention      │
│    Output: Structured ModelHypothesis JSON                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Adversarial Critique Agent                               │
│    Scrutinizes: Parameter over-fitting (e.g. Richards ν)     │
│    Evaluates: Residual normality and confidence bounds      │
│    Output: CritiqueReport (Score, Flaws, Concurrence)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Arbitration Engine                                       │
│    If Critique Concurs: Approve Selected Model              │
│    If Critique Rejects: Fallback to Minimum-AIC Parsimony   │
│    Output: ArbitrationDecision JSON                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Deterministic Guardrails & Hallucination Filter          │
│    Hard rules: μ_max > 0, λ ≥ 0, Non-negative, |Δparam| ≤ 5%│
│    Output: GuardrailAuditReport & Cryptographic Hash        │
└─────────────────────────────────────────────────────────────┘
```
