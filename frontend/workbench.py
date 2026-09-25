"""Presentation components adapted from Final_Project/scripts/31_streamlit_inference_app.py.

Preserves the original workbench styling and case evidence panels. The portfolio
entry point is app.py; frozen results are replayed without starting inference.
Source identity is recorded in data/showcase_manifest.json.
"""
from __future__ import annotations
import json
from typing import Any
import pandas as pd
import streamlit as st

CORE_COLUMNS = ["time_h", "X_obs", "S1_obs", "S2_obs"]
DISPLAY_SERIES = ["X_obs", "S1_obs", "S2_obs", "Ethanol_obs", "O2_obs", "CO2_obs"]
AGENT_FLOW = [
    ("Data Intake", "Validates columns, time ordering, missingness, and converts tables into RunData."),
    ("Diagnostic", "Computes reconstruction, residual, Stage 6, flux, GRU, and OOD evidence."),
    ("Hypothesis", "Produces structured raw and fittable hypotheses; LLM output is audited before use."),
    ("Fitting", "Fits ODE candidates and returns RMSE, AIC, BIC, parameters, and constraints."),
    ("Ranking", "Ranks the available ODE candidates, then applies configured rescue, projection, and harm checks."),
    ("Report", "Exports the final selected mechanism with warnings and audited evidence."),
]
EXPLAINERS = {
    "upload": (
        "The upload panel is a data-intake preview. It checks whether the time-series "
        "has the columns needed to become a pipeline RunData object and plots one run "
        "at a time when the file contains multiple mechanisms."
    ),
    "diagnostics": (
        "Diagnostics are evidence features, not final decisions. Residuals show where "
        "the fitted ODE misses the data; flux features add pathway-grounded clues; GRU "
        "scores summarize learned time-series evidence."
    ),
    "hypothesis": (
        "The raw hypothesis is the biological story proposed by the classifier or LLM. "
        "The fittable projection is the ODE mechanism that can actually be calibrated. "
        "These are kept separate so unsupported biological complexity does not silently "
        "become a model-selection decision."
    ),
    "pairwise": (
        "Pairwise critique compares mechanisms that are easy to confuse. It is advisory "
        "evidence: deterministic guards still check whether the preferred mechanism is "
        "supported by fits, residuals, flux diagnostics, and GRU scores."
    ),
    "fits": (
        "Each candidate row is an ODE fit. Lower RMSE means closer trajectory fit; lower "
        "AIC/BIC means a better fit after penalizing complexity. The selected mechanism "
        "comes from ranker and guard logic over these fitted candidates."
    ),
    "ranking": (
        "These gates are deterministic checks layered after fitting. They can rescue "
        "known failure modes, block over-complex promotions, or preserve a simpler model "
        "when biological support is weak."
    ),
    "report": (
        "The report is the human-readable evidence packet: final mechanism, hypothesis, "
        "supporting and contradictory evidence, pairwise critique, gates applied, and "
        "downloadable JSON/Markdown."
    ),
    "audit": (
        "The audit log is the complete machine-readable payload. It exists so every "
        "visible summary can be traced back to exact structured fields."
    ),
}

PAGE_CSS = """
<style>
    :root {
        --ns-bg: #0a0f0f;
        --ns-surface: #171918;
        --ns-surface-2: #1f221f;
        --ns-panel: #22261f;
        --ns-cream: #eef0d8;
        --ns-text: #d8dac6;
        --ns-muted: #aaaE9a;
        --ns-line: #3a4038;
        --ns-olive: #9ea681;
        --ns-olive-dark: #2d3324;
        --ns-cyan: #9fe8f2;
        --ns-green: #94d58d;
        --ns-warn: #d7b56d;
        --ns-bad: #e07676;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(102, 172, 163, 0.16), transparent 28rem),
            linear-gradient(180deg, #0a0f0f 0%, #0d1111 100%);
        color: var(--ns-text);
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1720px; }
    h1, h2, h3, h4, h5, h6 {
        color: var(--ns-cream) !important;
        letter-spacing: 0;
    }
    p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--ns-text);
    }
    [data-testid="stCaptionContainer"] {
        color: var(--ns-muted) !important;
    }
    section[data-testid="stSidebar"] {
        background: #111616;
        border-right: 1px solid var(--ns-line);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ns-text);
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: #23262c !important;
        border-color: #3d443c !important;
        color: var(--ns-cream) !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(23, 25, 24, 0.72);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: var(--ns-cream) !important;
    }
    button[kind="primary"], .stDownloadButton button {
        background: var(--ns-olive-dark) !important;
        color: var(--ns-cream) !important;
        border: 1px solid var(--ns-olive) !important;
        border-radius: 8px !important;
    }
    button[kind="primary"]:hover, .stDownloadButton button:hover {
        border-color: var(--ns-cyan) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--ns-line);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--ns-muted);
        background: transparent;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ns-cream) !important;
        background: var(--ns-surface-2);
        border-bottom: 2px solid var(--ns-olive);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #1b1e1d 0%, #161918 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
        color: var(--ns-cream) !important;
    }
    div[data-testid="stMetric"] * { color: var(--ns-cream) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--ns-muted) !important;
        font-weight: 650 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ns-cream) !important;
    }
    .hero {
        background:
            linear-gradient(135deg, rgba(37, 42, 37, 0.94) 0%, rgba(20, 22, 22, 0.98) 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
        color: var(--ns-cream);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.28);
    }
    .hero h3 { margin: 0 0 0.35rem 0; color: var(--ns-cream) !important; }
    .hero p { margin: 0; color: var(--ns-muted); }
    .intro-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.75rem 0 1.1rem 0;
    }
    .intro-card {
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        min-height: 132px;
        color: var(--ns-text);
    }
    .intro-card strong {
        display: block;
        margin-bottom: 0.35rem;
        color: var(--ns-cream);
    }
    .intro-card span {
        color: var(--ns-muted);
        font-size: 0.9rem;
        line-height: 1.42;
    }
    .summary-card {
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.95rem 1.05rem;
        min-height: 132px;
        color: var(--ns-text);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }
    .summary-card .summary-label {
        color: var(--ns-muted);
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .summary-card .summary-value {
        color: var(--ns-cream);
        font-size: 2.05rem;
        line-height: 1.05;
        font-weight: 520;
        margin-bottom: 0.55rem;
    }
    .summary-card .summary-help {
        color: var(--ns-muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }
    .metric-flow {
        display: grid;
        grid-template-columns: minmax(160px, 1fr) 34px minmax(180px, 1fr) 34px minmax(190px, 1fr);
        gap: 0.75rem;
        align-items: stretch;
        margin: 0.75rem 0 1.25rem 0;
    }
    .flow-card {
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        color: var(--ns-text);
        min-height: 146px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }
    .flow-card .flow-label {
        color: var(--ns-muted);
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .flow-card .flow-value {
        color: var(--ns-cream);
        font-size: 1.95rem;
        line-height: 1.05;
        font-weight: 560;
        margin-bottom: 0.45rem;
    }
    .flow-card .flow-help {
        color: var(--ns-muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }
    .flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--ns-olive);
        font-size: 1.7rem;
        font-weight: 700;
    }
    .flow-delta {
        display: inline-block;
        margin-top: 0.25rem;
        color: var(--ns-green);
        font-size: 0.8rem;
        font-weight: 700;
    }
    .agent-card {
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.75rem;
        min-height: 116px;
        color: var(--ns-text);
    }
    .agent-card strong { color: var(--ns-cream); }
    .agent-card span { color: var(--ns-muted); font-size: 0.86rem; }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        margin: 0.12rem 0.18rem 0.12rem 0;
        font-size: 0.76rem;
        font-weight: 650;
        border: 1px solid transparent;
    }
    .badge-good { color: #dff6d5; background: #25472b; border-color: #5f8f58; }
    .badge-idle { color: var(--ns-cream); background: var(--ns-olive-dark); border-color: #596047; }
    .badge-warn { color: #ffe4a8; background: #4b3820; border-color: #9e7d42; }
    .badge-bad { color: #ffd7d7; background: #512525; border-color: #9d4b4b; }
    .small-muted { color: var(--ns-muted); font-size: 0.88rem; }
    .decision-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        align-items: center;
        margin: 0.35rem 0 0.8rem 0;
    }
    .score-row {
        display: grid;
        grid-template-columns: minmax(140px, 220px) minmax(120px, 1fr) 70px;
        gap: 0.65rem;
        align-items: center;
        margin: 0.35rem 0;
        color: var(--ns-text);
    }
    .score-track {
        height: 0.55rem;
        background: #2a2e2b;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--ns-line);
    }
    .score-fill {
        height: 100%;
        background: var(--ns-cyan);
        border-radius: 999px;
    }
    .score-fill.strong { background: var(--ns-green); }
    .score-fill.warn { background: var(--ns-warn); }
    .failure-card {
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
        color: var(--ns-text);
    }
    .failure-card h4 {
        margin: 0 0 0.45rem 0;
        font-size: 1rem;
        color: var(--ns-cream);
    }
    .failure-grid {
        display: grid;
        grid-template-columns: minmax(150px, 0.42fr) 1fr;
        gap: 0.45rem 0.8rem;
        font-size: 0.9rem;
    }
    .failure-grid div:nth-child(odd) {
        color: var(--ns-muted);
        font-weight: 700;
    }
    .failure-grid div:nth-child(even) {
        color: var(--ns-text);
        overflow-wrap: anywhere;
        white-space: normal;
    }
    .guide-list {
        display: grid;
        gap: 0.55rem;
        margin-top: 0.55rem;
    }
    .guide-row {
        display: grid;
        grid-template-columns: minmax(180px, 0.32fr) 1fr;
        gap: 0.8rem;
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
    }
    .guide-key {
        color: var(--ns-cream);
        font-weight: 750;
        overflow-wrap: anywhere;
    }
    .guide-value {
        color: var(--ns-muted);
        line-height: 1.38;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .hierarchy-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.8rem 0 1rem 0;
    }
    .hierarchy-card {
        background: linear-gradient(180deg, #1b1e1d 0%, #151817 100%);
        border: 1px solid var(--ns-line);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 132px;
    }
    .hierarchy-card .step {
        display: inline-block;
        color: var(--ns-cream);
        background: var(--ns-olive-dark);
        border: 1px solid #596047;
        border-radius: 999px;
        padding: 0.15rem 0.5rem;
        font-size: 0.76rem;
        font-weight: 750;
        margin-bottom: 0.45rem;
    }
    .hierarchy-card strong {
        display: block;
        color: var(--ns-cream);
        margin-bottom: 0.35rem;
    }
    .hierarchy-card span {
        display: block;
        color: var(--ns-muted);
        font-size: 0.84rem;
        line-height: 1.38;
    }
    @media (max-width: 900px) {
        .intro-grid { grid-template-columns: 1fr; }
        .metric-flow { grid-template-columns: 1fr; }
        .flow-arrow { min-height: 18px; transform: rotate(90deg); }
        .failure-grid { grid-template-columns: 1fr; }
        .hierarchy-grid { grid-template-columns: 1fr; }
        .guide-row { grid-template-columns: 1fr; }
    }
</style>
"""


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    val = _safe_float(value)
    if val is not None:
        return f"{val:.{digits}f}"
    if value is None or value == "":
        return "n/a"
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _short_json(value: Any, limit: int = 160) -> str:
    if value in (None, "", [], {}):
        return ""
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return "n/a"
    return str(value)


def _arrow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == "object":
            safe[col] = safe[col].map(_display_value)
    return safe


def show_dataframe(df: pd.DataFrame, **kwargs: Any) -> None:
    safe = _arrow_safe_frame(df)
    try:
        st.dataframe(safe, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(safe, width="stretch", **kwargs)


def badge(label: str, kind: str = "idle") -> str:
    css_kind = {
        "good": "badge-good",
        "idle": "badge-idle",
        "warn": "badge-warn",
        "bad": "badge-bad",
    }.get(kind, "badge-idle")
    return f'<span class="badge {css_kind}">{label}</span>'


def summary_card(label: str, value: Any, help_text: str) -> str:
    return f"""
    <div class="summary-card">
      <div class="summary-label">{label}</div>
      <div class="summary-value">{value}</div>
      <div class="summary-help">{help_text}</div>
    </div>
    """




def explain(key: str, title: str = "What this means") -> None:
    text = EXPLAINERS.get(key)
    if not text:
        return
    with st.expander(title, expanded=False):
        st.write(text)


def render_project_introduction() -> None:
    st.markdown(
        """
        <div class="intro-grid">
          <div class="intro-card">
            <strong>Project Aim</strong>
            <span>
              Build an auditable architecture for bioprocessing where LLM hypotheses
              are constrained by mechanistic models, diagnostics, neural evidence,
              and explicit guardrails.
            </span>
          </div>
          <div class="intro-card">
            <strong>Why It Matters</strong>
            <span>
              Standalone LLM explanations can sound plausible while violating
              biochemical or physical logic. This workflow grounds each claim in
              time-series evidence, ODE fits, diagnostics, and constraints.
            </span>
          </div>
          <div class="intro-card">
            <strong>Case Study Input</strong>
            <span>
              Microbial growth time-series provide the demonstration data. The same
              guarded pattern is intended as a reusable design for bioprocess
              hypothesis validation.
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )








def validate_uploaded_timeseries(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    checks: list[str] = []
    warnings: list[str] = []

    missing = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing:
        warnings.append(f"Missing required columns: {', '.join(missing)}")
    else:
        checks.append("Required core columns present")

    if "time_h" in df.columns:
        time = pd.to_numeric(df["time_h"], errors="coerce")
        if time.isna().any():
            warnings.append("time_h contains non-numeric values")
        elif not (all(group.is_monotonic_increasing for _, group in time.groupby(df["run_id"], dropna=False)) if "run_id" in df else time.is_monotonic_increasing):
            warnings.append("time_h is not sorted in increasing order")
        else:
            checks.append("time_h is numeric and sorted")

    obs_cols = [col for col in df.columns if col.endswith("_obs")]
    if obs_cols:
        missingness = df[obs_cols].isna().mean().sort_values(ascending=False)
        worst_col = str(missingness.index[0])
        worst = float(missingness.iloc[0])
        checks.append(f"Observed channels found: {len(obs_cols)}")
        if worst > 0:
            warnings.append(f"Highest missingness: {worst_col} = {worst:.1%}")
    else:
        warnings.append("No *_obs measurement columns found")

    if "run_id" in df.columns:
        checks.append(f"Rows cover {df['run_id'].nunique()} run_id value(s)")

    return checks, warnings




def render_agent_flow(row: dict[str, Any] | None = None) -> None:
    states = {}
    if row:
        states = {
            "Data Intake": "checked",
            "Diagnostic": "checked" if row.get("diagnostic_summary") else "pending",
            "Hypothesis": _as_dict(row.get("llm_hypothesis")).get("source") or "rules",
            "Fitting": f"{len(_as_list(row.get('ranking')))} fits",
            "Ranking": row.get("best_mechanism") or "n/a",
            "Report": "ready",
        }
    cols = st.columns(3)
    for idx, (name, role) in enumerate(AGENT_FLOW):
        with cols[idx % 3]:
            state = states.get(name, "ready")
            st.markdown(
                f"""
                <div class="agent-card">
                  <strong>{name} Agent</strong><br>
                  <span>{role}</span><br><br>
                  {badge(str(state), "good" if state not in ("pending", "n/a") else "idle")}
                </div>
                """,
                unsafe_allow_html=True,
            )


def applied_gate_badges(row: dict[str, Any]) -> str:
    keys = [
        "stage6_rescue",
        "baseline_like_guard",
        "switching_projection_guard",
        "conservative_selection_bridge",
        "diagnostic_selection_guard",
        "candidate_fit_completion",
        "pairwise_harm_guard",
    ]
    parts = []
    for key in keys:
        payload = _as_dict(row.get(key))
        if not payload:
            parts.append(badge(f"{key}: missing", "warn"))
            continue
        applied = bool(payload.get("applied"))
        label = key.replace("_", " ")
        detail = payload.get("rule_fired") or payload.get("blocked_reason") or payload.get("new_best")
        if detail:
            label = f"{label}: {detail}"
        parts.append(badge(label, "good" if applied else "idle"))
    return "".join(parts)


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    safe = _arrow_safe_frame(df).fillna("n/a")
    headers = [str(c) for c in safe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, rec in safe.iterrows():
        values = [str(rec[col]).replace("\n", " ") for col in safe.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_markdown(row: dict[str, Any]) -> str:
    llm = _as_dict(row.get("llm_hypothesis"))
    ranking = ranking_frame(row)
    gates = gates_frame(row)
    pairwise = _as_list(llm.get("pairwise_critiques"))
    lines = [
        f"# Mechanism Report: {row.get('run_id', 'run')}",
        "",
        f"- Selected mechanism: **{row.get('best_mechanism', 'n/a')}**",
        f"- LLM hypothesis: **{llm.get('hypothesis', 'n/a')}**",
        f"- Raw biological hypothesis: **{row.get('raw_hypothesis', 'n/a')}**",
        f"- Fittable projection: **{row.get('fittable_hypothesis', 'n/a')}**",
        f"- Evidence source: **{llm.get('source', 'n/a')}**",
        f"- Confidence: **{_fmt(llm.get('confidence'))}**",
        f"- Best RMSE: **{_fmt(row.get('best_rmse_total'))}**",
        f"- Best AIC: **{_fmt(row.get('best_aic'), digits=2)}**",
        "",
        "## Hypothesis Evidence",
        "### Evidence For",
        "\n".join(f"- {item}" for item in _as_list(llm.get("evidence_for"))) or "- n/a",
        "",
        "### Evidence Against",
        "\n".join(f"- {item}" for item in _as_list(llm.get("evidence_against"))) or "- n/a",
        "",
        "### Required Validation",
        "\n".join(f"- {item}" for item in _as_list(llm.get("required_validation"))) or "- n/a",
        "",
        "## Pairwise Critique",
        "\n\n".join(_pairwise_markdown(item) for item in pairwise) if pairwise else "No pairwise critique was recorded.",
        "",
        "## Deterministic Gate Status",
        frame_to_markdown(gates) if not gates.empty else "No gate payloads were recorded.",
        "",
        "## Candidate Ranking",
        frame_to_markdown(ranking) if not ranking.empty else "No ranking table was recorded.",
        "",
        "## Narrative",
        llm.get("critique") or "No narrative critique was recorded.",
    ]
    return "\n".join(lines)


def case_interpretation(row: dict[str, Any]) -> str:
    selected = row.get("best_mechanism") or "unknown"
    raw = row.get("raw_hypothesis") or "unknown"
    fittable = row.get("fittable_hypothesis") or "unknown"
    source = _as_dict(row.get("llm_hypothesis")).get("source") or "unknown"
    return (
        f"The system selected **{selected}** after checking the raw biological "
        f"hypothesis **{raw}** against the fittable surrogate **{fittable}**. "
        f"The hypothesis source was **{source}**, and deterministic gates/fits "
        "were applied before final selection."
    )


def correctness_badges(row: dict[str, Any]) -> str:
    parts = []
    if row.get("correct_selection") is not None:
        parts.append(
            badge(
                "saved projection correct" if row.get("correct_selection") else "saved projection mismatch",
                "good" if row.get("correct_selection") else "bad",
            )
        )
    if row.get("correct_hypothesis") is not None:
        parts.append(
            badge(
                "saved hypothesis correct" if row.get("correct_hypothesis") else "saved hypothesis mismatch",
                "good" if row.get("correct_hypothesis") else "warn",
            )
        )
    truth = row.get("true_mechanism") or "baseline"
    if truth not in (None, "n/a"):
        parts.append(badge(f"truth: {truth}", "idle"))
    return "".join(parts)


def mechanism_evidence_frame(row: dict[str, Any]) -> pd.DataFrame:
    diag = _as_dict(row.get("diagnostic_summary"))
    evidence = _as_dict(diag.get("mechanism_evidence"))
    if not evidence:
        keys = [
            ("maintenance", "maintenance_respiration_score"),
            ("product_inhibition", "product_inhibition_score"),
            ("oxygen_limitation", "oxygen_limitation_score"),
            ("yield_drift", "yield_drift_score"),
            ("switching", "switching_score"),
            ("death_decay", "death_decay_score"),
            ("lag_adaptation", "lag_adaptation_score"),
        ]
        evidence = {name: diag.get(key) for name, key in keys if diag.get(key) is not None}
    records = [
        {"Mechanism": key, "Score": _safe_float(value)}
        for key, value in evidence.items()
        if _safe_float(value) is not None
    ]
    return pd.DataFrame(records).sort_values("Score", ascending=False) if records else pd.DataFrame()


def render_mechanism_evidence(row: dict[str, Any]) -> None:
    df = mechanism_evidence_frame(row)
    if df.empty:
        st.caption("No mechanism evidence scores were recorded for this run.")
        return
    for _, rec in df.head(8).iterrows():
        score = float(rec["Score"])
        kind = "strong" if score >= 0.7 else ("warn" if score >= 0.4 else "")
        st.markdown(
            f"""
            <div class="score-row">
              <div><strong>{rec['Mechanism']}</strong></div>
              <div class="score-track"><div class="score-fill {kind}" style="width:{max(0, min(1, score)) * 100:.1f}%"></div></div>
              <div>{score:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )




def _ranking_summary(row: dict[str, Any]) -> str:
    pieces = []
    for fit in _as_list(row.get("ranking"))[:3]:
        mech = fit.get("mechanism") or fit.get("mechanism_name") or "unknown"
        rmse = _fmt(fit.get("rmse_total"))
        bic = _fmt(fit.get("bic"), digits=2)
        pieces.append(f"{mech}: RMSE {rmse}, BIC {bic}")
    return "; ".join(pieces)


def _guard_summary(row: dict[str, Any]) -> str:
    names = [
        "diagnostic_selection_guard",
        "candidate_fit_completion",
        "conservative_selection_bridge",
        "stage6_rescue",
        "baseline_like_guard",
        "switching_projection_guard",
        "pairwise_harm_guard",
    ]
    pieces = []
    for name in names:
        payload = _as_dict(row.get(name))
        if not payload:
            continue
        if payload.get("applied"):
            detail = payload.get("rule_fired") or payload.get("new_best") or payload.get("target") or "applied"
            pieces.append(f"{name}: {detail}")
    raw_rules = _as_list(_as_dict(row.get("llm_hypothesis")).get("guardrail_applied_rules"))
    if raw_rules:
        pieces.append("raw_hypothesis_guard: " + ", ".join(str(x) for x in raw_rules))
    return "; ".join(pieces) or "No applied guard recorded"


def _why_selected_summary(row: dict[str, Any]) -> str:
    selected = row.get("best_mechanism") or "unknown"
    ranking = _as_list(row.get("ranking"))
    selected_fit = None
    for fit in ranking:
        mech = fit.get("mechanism") or fit.get("mechanism_name")
        if mech == selected:
            selected_fit = fit
            break
    fit_text = ""
    if selected_fit:
        fit_text = (
            f"The selected mechanism had RMSE {_fmt(selected_fit.get('rmse_total'))} "
            f"and BIC {_fmt(selected_fit.get('bic'), digits=2)} in the fitted candidate table."
        )
    else:
        fit_text = "The selected mechanism was produced by the ranking or guardrail layer."

    guard_text = _guard_summary(row)
    if guard_text == "No applied guard recorded":
        return f"{fit_text} No applied guard changed the final selection."
    return f"{fit_text} Applied decision logic: {guard_text}."


def _how_concluded_summary(row: dict[str, Any]) -> str:
    llm = _as_dict(row.get("llm_hypothesis"))
    diag = _as_dict(row.get("diagnostic_summary"))
    source = llm.get("source") or "rules"
    raw = row.get("raw_hypothesis") or "unknown"
    projection = row.get("fittable_hypothesis") or "unknown"
    scores = mechanism_evidence_frame(row).head(3)
    score_text = ""
    if not scores.empty:
        score_text = " Strongest diagnostic scores: " + ", ".join(
            f"{rec['Mechanism']}={float(rec['Score']):.3f}" for _, rec in scores.iterrows()
        ) + "."
    elif diag:
        score_text = " Diagnostic evidence was present but no mechanism score summary was available."
    return (
        f"The {source} hypothesis layer proposed raw mechanism '{raw}', which was mapped to "
        f"fittable mechanism '{projection}'. ODE candidates were fitted and ranked, then diagnostic "
        f"and guardrail checks audited the choice.{score_text}"
    )






def _pairwise_markdown(item: Any) -> str:
    if not isinstance(item, dict):
        return f"- {item}"
    pair = " vs ".join(str(x) for x in _as_list(item.get("pair"))) or "unknown pair"
    preferred = item.get("preferred", "unknown")
    for_pref = "\n".join(f"  - {x}" for x in _as_list(item.get("evidence_for_preferred"))) or "  - n/a"
    against = "\n".join(f"  - {x}" for x in _as_list(item.get("evidence_against_other"))) or "  - n/a"
    return (
        f"### {pair}\n"
        f"- Preferred: **{preferred}**\n"
        f"- Evidence for preferred:\n{for_pref}\n"
        f"- Evidence against other:\n{against}"
    )


def render_list_block(title: str, values: Any, empty: str = "No entries recorded.") -> None:
    st.markdown(f"**{title}**")
    captions = {
        "Evidence For": "Advisory support for the proposed raw biological hypothesis. It helps explain the LLM choice but does not decide the final mechanism alone.",
        "Evidence Against": "Advisory contradictions or competing signals. Strong contradiction can trigger guards or push the final selector toward another fit.",
        "Required Validation": "Required checks before the hypothesis can be trusted. These connect the LLM proposal to constraints, ODE fits, and physical admissibility.",
        "Evidence for preferred": "Advisory signals supporting the preferred side of this pairwise comparison.",
        "Evidence against other": "Advisory signals weakening the non-preferred mechanism in this comparison.",
    }
    if title in captions:
        st.caption(captions[title])
    items = _as_list(values)
    if not items:
        st.caption(empty)
        return
    for item in items:
        st.markdown(f"- {item}")


def render_hypothesis_hierarchy(row: dict[str, Any]) -> None:
    diag_guard = _as_dict(row.get("diagnostic_selection_guard"))
    candidate_completion = _as_dict(row.get("candidate_fit_completion"))
    guard_state = "Applied" if diag_guard.get("applied") else "Checked"
    completion_state = "Applied" if candidate_completion.get("applied") else "Checked"
    st.markdown(
        f"""
        <div class="hierarchy-grid">
          <div class="hierarchy-card">
            <div class="step">1. Proposal</div>
            <strong>Structured raw hypothesis</strong>
            <span>The LLM, frozen classifier, or rule layer proposes a biological mechanism label and records its available support.</span>
          </div>
          <div class="hierarchy-card">
            <div class="step">2. Advisory Evidence</div>
            <strong>For / against / pairwise critique</strong>
            <span>These fields explain the hypothesis and common confusions. They are evidence notes, not final authority.</span>
          </div>
          <div class="hierarchy-card">
            <div class="step">3. Required Checks</div>
            <strong>Validation and fittable projection</strong>
            <span>The raw label must map to a fittable ODE candidate and pass required checks such as constraints and fit support.</span>
          </div>
          <div class="hierarchy-card">
            <div class="step">4. Override Layer</div>
            <strong>{guard_state} guards / {completion_state} fit completion</strong>
            <span>ODE ranking, diagnostics, candidate completion, and guardrails can override or preserve the LLM proposal.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pairwise_critiques(pairwise: list[Any]) -> None:
    if not pairwise:
        st.caption("No pairwise critique was produced for this run.")
        return
    for idx, item in enumerate(pairwise, start=1):
        if not isinstance(item, dict):
            st.write(item)
            continue
        pair = " vs ".join(str(x) for x in _as_list(item.get("pair"))) or f"Pair {idx}"
        preferred = item.get("preferred", "unknown")
        with st.container(border=True):
            top = st.columns([1.4, 0.8])
            top[0].markdown(f"**{pair}**")
            top[1].markdown(f"Preferred: **{preferred}**")
            cols = st.columns(2)
            with cols[0]:
                render_list_block("Evidence for preferred", item.get("evidence_for_preferred"))
            with cols[1]:
                render_list_block("Evidence against other", item.get("evidence_against_other"))


def render_hypothesis_evidence(llm: dict[str, Any], row: dict[str, Any], gap: dict[str, Any]) -> None:
    source = str(llm.get("source") or "unknown")
    classifier = _as_dict(llm.get("clf_result"))
    is_classifier = source == "frozen_classifier" or bool(classifier)

    summary_cols = st.columns(5)
    summary_cols[0].metric("Hypothesis", llm.get("hypothesis") or row.get("raw_hypothesis") or "n/a")
    summary_cols[1].metric("Raw hypothesis", row.get("raw_hypothesis") or "n/a")
    summary_cols[2].metric("Fittable projection", row.get("fittable_hypothesis") or "n/a")
    summary_cols[3].metric("Source", llm.get("source") or "n/a")
    summary_cols[4].metric("Confidence", _fmt(llm.get("confidence")))

    render_hypothesis_hierarchy(row)

    st.markdown("**Hypothesis reasoning**")
    if is_classifier:
        st.caption(
            "This proposal came from the frozen classifier, not an LLM critique. The classifier "
            "records probabilities and trust checks; downstream ODE fitting, diagnostics, and "
            "guardrails still determine the final mechanism."
        )
        st.write(llm.get("reasoning") or "No classifier reasoning summary was recorded.")
    else:
        st.caption(
            "Human-readable reasoning from the hypothesis layer. This explains the proposal, "
            "but downstream ODE fitting, diagnostics, and guardrails decide whether it is accepted."
        )
        st.write(llm.get("critique") or llm.get("reasoning") or "No narrative reasoning was recorded.")

    evidence_cols = st.columns(3)
    if is_classifier:
        distribution = _as_dict(llm.get("raw_distribution"))
        hypothesis = str(llm.get("hypothesis") or row.get("raw_hypothesis") or "unknown")
        supporting = []
        if hypothesis in distribution:
            supporting.append(f"Classifier probability for {hypothesis}: {float(distribution[hypothesis]):.4f}")
        if classifier.get("trusted") is not None:
            supporting.append(f"Classifier trust gate passed: {bool(classifier.get('trusted'))}")

        competing = sorted(
            ((name, score) for name, score in distribution.items() if name != hypothesis),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:3]
        against = [f"Competing probability for {name}: {float(score):.4f}" for name, score in competing]

        checks = []
        if classifier.get("entropy") is not None:
            checks.append(f"Classifier entropy: {float(classifier['entropy']):.4f}")
        if classifier.get("is_entropy_ambiguous") is not None:
            checks.append(f"Entropy ambiguity flag: {bool(classifier.get('is_entropy_ambiguous'))}")
        if classifier.get("is_dist_ood") is not None:
            checks.append(f"Out-of-distribution flag: {bool(classifier.get('is_dist_ood'))}")
        if llm.get("is_supported_by_registry") is not None:
            checks.append(f"Mechanism registered in the allowed hypothesis space: {bool(llm.get('is_supported_by_registry'))}")

        with evidence_cols[0]:
            render_list_block("Evidence For", supporting, "No classifier support values were stored.")
        with evidence_cols[1]:
            render_list_block("Evidence Against", against, "No competing class probabilities were stored.")
        with evidence_cols[2]:
            render_list_block("Required Validation", checks, "No classifier trust checks were stored.")
    else:
        with evidence_cols[0]:
            render_list_block("Evidence For", llm.get("evidence_for"))
        with evidence_cols[1]:
            render_list_block("Evidence Against", llm.get("evidence_against"))
        with evidence_cols[2]:
            render_list_block("Required Validation", llm.get("required_validation"))

    with st.expander("Hypothesis gap and distributions", expanded=False):
        st.json(gap, expanded=False)


def render_uploaded_csv(uploaded_file: Any) -> None:
    st.subheader("Upload CSV")
    st.caption(
        "Preview the measurements in your CSV. Uploads do not run inference or change the frozen results."
    )
    explain("upload")

    if uploaded_file is None:
        st.info("Upload a CSV with columns such as time_h, X_obs, S1_obs, and S2_obs.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    checks, warnings = validate_uploaded_timeseries(df)
    chart_df = df
    if "run_id" in df.columns and df["run_id"].nunique() > 1:
        run_ids = sorted(df["run_id"].dropna().astype(str).unique())
        chosen_run = st.selectbox("Preview uploaded run", run_ids, key="uploaded_preview_run")
        chart_df = df[df["run_id"].astype(str) == chosen_run].copy()
        st.caption(f"Uploaded file contains {len(run_ids)} runs. Previewing: {chosen_run}")

    left, right = st.columns([1.2, 0.8])
    with left:
        chart_cols = [col for col in DISPLAY_SERIES if col in chart_df.columns]
        if "time_h" in chart_df.columns and chart_cols:
            plot_df = chart_df[["time_h", *chart_cols]].copy()
            plot_df["time_h"] = pd.to_numeric(plot_df["time_h"], errors="coerce")
            st.line_chart(plot_df.set_index("time_h"))
        else:
            show_dataframe(chart_df.head(25))

    with right:
        st.markdown("**Data Intake Checks**")
        for item in checks:
            st.success(item)
        for item in warnings:
            st.warning(item)
        st.markdown("**Preview**")
        show_dataframe(chart_df.head(8))


def ranking_frame(row: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx, fit in enumerate(_as_list(row.get("ranking")), start=1):
        success = fit.get("success")
        records.append(
            {
                "Rank": idx,
                "Mechanism": fit.get("mechanism") or fit.get("mechanism_name"),
                "RMSE": _safe_float(fit.get("rmse_total")),
                "AIC": _safe_float(fit.get("aic")),
                "BIC": _safe_float(fit.get("bic")),
                "Score": _safe_float(fit.get("composite_score", fit.get("score"))),
                "Constraints": "pass" if success is True else ("fail" if success is False else "n/a"),
            }
        )
    return pd.DataFrame(records)


def diagnostics_frame(row: dict[str, Any]) -> pd.DataFrame:
    diag = _as_dict(row.get("diagnostic_summary"))
    mixed = _as_dict(diag.get("mixed_mechanism_features"))
    flux = _as_dict(diag.get("extracellular_flux_features"))
    gru = _as_dict(diag.get("gru_evidence_scores"))

    records: list[dict[str, Any]] = []
    for section, mapping in [
        ("core", diag),
        ("mechanism_evidence", _as_dict(diag.get("mechanism_evidence"))),
        ("mixed_mechanism_features", mixed),
        ("extracellular_flux_features", flux),
        ("gru_evidence_scores", gru),
    ]:
        for key, value in mapping.items():
            if isinstance(value, (dict, list)):
                continue
            records.append({"Section": section, "Metric": key, "Value": _display_value(value)})
    return pd.DataFrame(records)


def render_diagnostic_section_guide() -> None:
    st.markdown("**Diagnostic Section Guide**")
    st.caption("These labels group the diagnostic table into evidence sources.")
    guide = [
        (
            "core",
            "Top-level run diagnostics such as RMSE, residual source, phase summaries, fitted mechanism, and scalar scores.",
        ),
        (
            "mechanism_evidence",
            "Continuous support scores for each mechanism. These summarize diagnostic evidence but do not directly select the final model.",
        ),
        (
            "mixed_mechanism_features",
            "Rule-style indicators for mixed or compound mechanisms, such as maintenance plus inhibition or switching plus inhibition.",
        ),
        (
            "extracellular_flux_features",
            "Metabolite and flux-derived signals, including product accumulation, oxygen limitation proxies, overflow behaviour, and yield drift clues.",
        ),
        (
            "gru_evidence_scores",
            "Neural time-series evidence scores from the GRU model. These are advisory signals used by diagnostics and guardrails.",
        ),
    ]
    rows = "".join(
        f'<div class="guide-row"><div class="guide-key">{section}</div>'
        f'<div class="guide-value">{description}</div></div>'
        for section, description in guide
    )
    st.markdown(f'<div class="guide-list">{rows}</div>', unsafe_allow_html=True)


def gates_frame(row: dict[str, Any]) -> pd.DataFrame:
    gate_keys = [
        "stage6_rescue",
        "baseline_like_guard",
        "switching_projection_guard",
        "conservative_selection_bridge",
        "diagnostic_selection_guard",
        "candidate_fit_completion",
        "pairwise_harm_guard",
    ]
    records = []
    for key in gate_keys:
        payload = _as_dict(row.get(key))
        if not payload:
            records.append({"Gate": key, "State": "missing", "Details": ""})
            continue
        applied = payload.get("applied")
        state = "applied" if applied else "idle"
        detail_parts = []
        for detail_key in ["rule_fired", "blocked_reason", "target", "reason", "new_best"]:
            if payload.get(detail_key) not in (None, "", []):
                detail_parts.append(f"{detail_key}={payload.get(detail_key)}")
        records.append({"Gate": key, "State": state, "Details": "; ".join(detail_parts)})
    return pd.DataFrame(records)


def render_case(row: dict[str, Any], report_context: str = "") -> None:
    llm = _as_dict(row.get("llm_hypothesis"))
    gap = _as_dict(row.get("hypothesis_gap"))

    st.subheader("Final Mechanism Report")
    st.caption(
        "Structured hypothesis evidence is checked by deterministic fitting, ranking, "
        "rescue, bridge, and harm-guard logic before the final mechanism is shown."
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Selected mechanism", row.get("best_mechanism") or "n/a")
    metric_cols[1].metric("Raw hypothesis", row.get("raw_hypothesis") or "n/a")
    metric_cols[2].metric("Fittable projection", row.get("fittable_hypothesis") or "n/a")
    metric_cols[3].metric("Evidence source", llm.get("source") or "n/a")
    metric_cols[4].metric("Confidence", _fmt(llm.get("confidence")))
    if row.get("correct_selection") is None:
        selection_status = "unknown"
    else:
        selection_status = "correct" if row.get("correct_selection") else "incorrect"
    metric_cols[5].metric("Saved projection status", selection_status)
    st.markdown(
        f'<div class="decision-strip">{correctness_badges(row)}</div>',
        unsafe_allow_html=True,
    )

    quality_cols = st.columns(4)
    quality_cols[0].metric("Best RMSE", _fmt(row.get("best_rmse_total")))
    quality_cols[1].metric("Best AIC", _fmt(row.get("best_aic"), digits=2))
    quality_cols[2].metric("Baseline RMSE", _fmt(row.get("baseline_rmse")))
    quality_cols[3].metric("Arbitration", "not recorded" if "arbitration_triggered" not in row else "yes" if row["arbitration_triggered"] else "no")
    st.info(case_interpretation(row))
    st.markdown(applied_gate_badges(row), unsafe_allow_html=True)

    tab_upload, tab_diag, tab_hyp, tab_fits, tab_rank, tab_report, tab_audit = st.tabs(
        [
            "Upload",
            "Diagnostics",
            "Hypothesis",
            "Model Fits",
            "Ranking",
            "Final Report",
            "Audit Log",
        ]
    )

    with tab_upload:
        st.info("Use the Synthetic data page to preview bundled samples or upload a CSV. Samples are not automatically linked to this saved decision.")
        explain("upload")
        st.json(
            {
                "run_id": row.get("run_id"),
                "scenario": row.get("scenario"),
                "trigger_reasons": row.get("trigger_reasons", []),
            },
            expanded=False,
        )

    with tab_diag:
        explain("diagnostics")
        left, right = st.columns([0.9, 1.1])
        with left:
            st.markdown("**Mechanism Evidence Scores**")
            st.caption("Continuous diagnostic support. High scores inform guardrails; they do not directly replace model ranking.")
            render_mechanism_evidence(row)
        with right:
            diag_df = diagnostics_frame(row)
            if diag_df.empty:
                st.warning("No diagnostic summary fields were found for this run.")
            else:
                show_dataframe(diag_df, hide_index=True)
                render_diagnostic_section_guide()

    with tab_hyp:
        explain("hypothesis")
        render_hypothesis_evidence(llm, row, gap)
        st.markdown("**Pairwise Critique**")
        explain("pairwise")
        render_pairwise_critiques(_as_list(llm.get("pairwise_critiques")))

    with tab_fits:
        explain("fits")
        rank_df = ranking_frame(row)
        if rank_df.empty:
            st.warning("No ranked fit table was found for this run.")
        else:
            show_dataframe(rank_df, hide_index=True)

    with tab_rank:
        explain("ranking")
        st.markdown("**Deterministic Gates**")
        show_dataframe(gates_frame(row), hide_index=True)
        st.markdown("**Agent Contract**")
        st.code(
            "HypothesisAgent -> structured raw_hypothesis + fittable_hypothesis\n"
            "FittingAgent -> list[FitResult]\n"
            "RankingAgent -> selected mechanism after ModelRanker and configured gates\n"
            "ReportAgent -> readable report over audited result",
            language="text",
        )

    with tab_report:
        explain("report")
        critique = llm.get("critique") or "No narrative critique was recorded."
        st.markdown("**Narrative**")
        st.write(critique)
        report_cols = st.columns(2)
        with report_cols[0]:
            render_list_block("Evidence For", llm.get("evidence_for"))
            render_list_block("Required Validation", llm.get("required_validation"))
        with report_cols[1]:
            render_list_block("Evidence Against", llm.get("evidence_against"))
            st.markdown("**Pairwise Summary**")
            render_pairwise_critiques(_as_list(llm.get("pairwise_critiques")))
        markdown_report = report_markdown(row)
        if report_context:
            markdown_report = report_context + "\n\n" + markdown_report
        st.download_button(
            "Download Markdown report",
            data=markdown_report,
            file_name=f"{row.get('run_id', 'mechanism_report')}.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download JSON report",
            data=json.dumps(row, indent=2),
            file_name=f"{row.get('run_id', 'mechanism_report')}.json",
            mime="application/json",
        )

    with tab_audit:
        explain("audit")
        st.json(row, expanded=False)






