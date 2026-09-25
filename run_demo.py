#!/usr/bin/env python3
"""
End-to-End Synthetic Demonstration Runner
=========================================
Runs a full cycle of the Neuro-Symbolic Scientific Model Selection framework
on a synthesized microbial growth trajectory, generating full provenance,
ODE fits, neural evidence, structured reasoning, and guardrail audit reports.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.ode_fitting import ODEFitter
from src.gru_model import (
    SyntheticTrajectoryGenerator,
    TemporalEvidenceEncoder,
)
from src.llm_reasoning import StructuredReasoningEngine
from src.guardrails import DeterministicGuardrails
from src.audit_logger import AuditLogger


def run_pipeline(output_json_path: str = "examples/synthetic-example.json"):
    print("=" * 78)
    print(" NEURO-SYMBOLIC AI FRAMEWORK FOR SCIENTIFIC MODEL SELECTION")
    print(" Illustrative demo — untrained GRU and mock reasoning")
    print("=" * 78)

    print("Not a reproduction of the reported Final_Project research results.")

    # 1. Generate Controlled Synthetic Trajectory
    print("\n[Step 1] Synthesizing microbial growth trajectory (Gompertz ground truth)...")
    traj = SyntheticTrajectoryGenerator.generate_single_trajectory(
        model_name="Gompertz",
        duration=48.0,
        num_points=49,
        snr_db=28.0,
        seed=2026,
    )
    t = traj["time"]
    y = traj["od600_noisy"]
    print(f"  • Ground Truth Model:  {traj['true_model']}")
    print(f"  • True Parameters:     {traj['ground_truth_params']}")
    print(f"  • Sampling Points:     {len(t)} points across {traj['duration']} hours (SNR: {traj['snr_db']} dB)")

    # 2. Mechanistic ODE Non-linear Regression
    print("\n[Step 2] Executing Mechanistic ODE Fitting Suite (Gompertz, Logistic, Richards, Baranyi)...")
    start_t = time.perf_counter()
    fitter = ODEFitter()
    selection_metrics = fitter.compare_models(t, y)
    ode_fits_dict = {k: v.to_dict() for k, v in selection_metrics.results.items()}

    for name, res in selection_metrics.results.items():
        print(f"  • {name:10s} | AICc: {res.aicc:7.2f} | BIC: {res.bic:7.2f} | R²: {res.r_squared:.4f} | RSS: {res.rss:.4f}")

    print(f"  -> Statistical Selection: Best by AICc = {selection_metrics.best_model_by_aic}")

    # 3. Deep Temporal Neural Feature Extraction
    print("\n[Step 3] Extracting Temporal Evidence via Bidirectional GRU with Attention Pooling...")
    encoder = TemporalEvidenceEncoder()
    neural_evidence = encoder.extract_evidence(t, y)
    print(f"  • Neural Prior:        {neural_evidence['predicted_model_prior']} (Confidence: {neural_evidence['confidence']:.3f})")
    print(f"  • Peak Attention Time: {neural_evidence['peak_attention_time_hours']:.2f} h (untrained attention; not a validated transition)")
    print(f"  • Class Probabilities: {neural_evidence['class_probabilities']}")

    # 4. Structured LLM Scientific Reasoning & Hypothesis Generation
    print("\n[Step 4] Formulating Structured Scientific Hypothesis & Reasoning Payload...")
    reasoning_engine = StructuredReasoningEngine(mode="mock_deterministic")
    prompt = reasoning_engine.build_prompt(ode_fits_dict, neural_evidence, {"duration_hours": 48.0})
    hypothesis = reasoning_engine.generate_hypothesis(ode_fits_dict, neural_evidence, {"duration_hours": 48.0})
    print(f"  • Selected Model:      {hypothesis.selected_model}")
    print(f"  • Confidence:          {hypothesis.confidence_score:.3f}")
    print(f"  • Mechanistic Rationale:\n    \"{hypothesis.mechanistic_rationale}\"")

    # 5. Multi-Agent Critique & Arbitration
    print("\n[Step 5] Multi-Agent Adversarial Critique & Arbitration...")
    critique = reasoning_engine.critique_hypothesis(hypothesis, ode_fits_dict, neural_evidence)
    arbitration = reasoning_engine.arbitrate(hypothesis, critique, ode_fits_dict)
    print(f"  • Critique Score:      {critique.critique_score:.2f} / 1.00 (Concurs: {critique.concurs_with_selection})")
    print(f"  • Arbitration Verdict: {arbitration.final_selected_model} (Consensus: {arbitration.consensus_reached})")

    # 6. Deterministic Guardrails & Physical Consistency Checks
    print("\n[Step 6] Running Deterministic Guardrails & Hallucination Filter...")
    guardrails = DeterministicGuardrails(hallucination_tolerance=0.05)
    selected_fit = selection_metrics.results[arbitration.final_selected_model]
    guard_report = guardrails.audit(
        selected_model=arbitration.final_selected_model,
        fitted_params=selected_fit.parameters,
        fitted_r_squared=selected_fit.r_squared,
        llm_cited_params=hypothesis.cited_parameters,
    )
    print(f"  • Overall Status:      {guard_report.overall_status.value}")
    print(f"  • Checks Passed:       {guard_report.passed_checks} / {guard_report.total_checks}")
    for rule in guard_report.rule_results:
        status_symbol = "✓" if rule.status.value == "PASSED" else "✗"
        print(f"    [{status_symbol}] {rule.rule_name:28s} -> {rule.details}")

    # 7. Provenance & Cryptographic Audit Logging
    print("\n[Step 7] Generating Cryptographic Audit Provenance Record...")
    elapsed_total_ms = (time.perf_counter() - start_t) * 1000.0
    audit_logger = AuditLogger(log_filepath="examples/audit_trail.jsonl")
    record = audit_logger.create_record(
        run_id="DEMO-RUN-2026-09-SYNTH-01",
        raw_trajectory={"time": t.tolist(), "od600_noisy": y.tolist()},
        ode_fits=ode_fits_dict,
        neural_evidence=neural_evidence,
        llm_reasoning=hypothesis.model_dump(),
        guardrail_report=guard_report.to_dict(),
        selected_model=arbitration.final_selected_model,
        execution_time_ms=elapsed_total_ms,
        metadata={
            "artifact_kind": "illustrative_demo_not_research_evidence",
            "reasoning_mode": "mock_deterministic",
            "neural_weights": "randomly_initialized",
            "python_version": sys.version.split()[0],
            "dataset_type": "controlled_synthetic_microbial_growth",
        },
    )
    print(f"  • Run ID:              {record.run_id}")
    print(f"  • SHA-256 Digest:      {record.final_decision_hash}")
    print(f"  • Latency:             {record.execution_time_ms:.2f} ms")

    # 8. Save structured synthetic example JSON
    full_example_payload = {
        "metadata": {
            "title": "Neuro-Symbolic Scientific Model Selection — Synthetic Case Study",
            "project": "Separate portfolio illustration",
            "artifact_kind": "illustrative_demo_not_research_evidence",
            "reasoning_mode": "mock_deterministic",
            "neural_weights": "randomly_initialized",
            "data_governance_notice": "Synthetically generated microbial growth trajectory. Contains zero proprietary or non-public experimental data.",
            "provenance": record.to_dict(),
        },
        "trajectory_data": {
            "time_hours": [round(float(x), 3) for x in t],
            "od600_observed": [round(float(x), 4) for x in y],
            "od600_clean_ground_truth": [round(float(x), 4) for x in traj["od600_clean"]],
            "ground_truth_model": traj["true_model"],
            "ground_truth_parameters": traj["ground_truth_params"],
        },
        "mechanistic_ode_fits": ode_fits_dict,
        "temporal_neural_evidence": neural_evidence,
        "structured_llm_reasoning": {
            "hypothesis": hypothesis.model_dump(),
            "critique": critique.model_dump(),
            "arbitration": arbitration.model_dump(),
        },
        "deterministic_guardrail_audit": guard_report.to_dict(),
    }

    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_example_payload, f, indent=2)

    print(f"\n[Success] Full synthetic case study saved to: {out_file.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    run_pipeline()
