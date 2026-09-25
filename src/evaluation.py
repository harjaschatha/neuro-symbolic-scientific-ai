"""Illustrative five-arm simulation; NOT empirical research evaluation.

Uses an untrained default GRU, mock reasoning, ground-truth-assisted random
predictions, injected parameter drift, and constructed hallucination counts.
Reported research evidence is exported by scripts/export_verified_results.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.ode_fitting import ODEFitter
from src.gru_model import (
    SyntheticTrajectoryGenerator,
    TemporalEvidenceEncoder,
    TemporalEvidenceGRU,
)
from src.llm_reasoning import StructuredReasoningEngine
from src.guardrails import DeterministicGuardrails, GuardrailStatus
from src.audit_logger import AuditLogger


class ArmType(str, Enum):
    ARM_1_PURE_ODE = "Arm 1: Pure Mechanistic ODE"
    ARM_2_PURE_NEURAL_GRU = "Arm 2: Pure Neural GRU"
    ARM_3_PURE_LLM_ZERO_SHOT = "Arm 3: Pure LLM Zero-Shot"
    ARM_4_HYBRID_UNCONSTRAINED = "Arm 4: Hybrid Unconstrained (No Guardrails)"
    ARM_5_NEURO_SYMBOLIC_GUARDED = "Arm 5: Full Guarded Neuro-Symbolic (Proposed)"


@dataclass
class ArmResult:
    arm_type: ArmType
    num_samples: int
    accuracy_percent: float
    mean_param_mape_percent: float
    hallucination_rate_percent: float
    mean_rss: float
    expected_calibration_error: float
    avg_latency_ms: float
    violations_caught: int


@dataclass
class EvaluationSummary:
    timestamp_utc: str
    total_trajectories_evaluated: int
    snr_db: float
    arm_results: Dict[str, ArmResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "evaluation_kind": "illustrative_simulation_not_research_results",
            "reasoning_mode": "mock_deterministic",
            "neural_weights": "randomly_initialized",
            "total_trajectories_evaluated": self.total_trajectories_evaluated,
            "snr_db": self.snr_db,
            "arm_results": {
                name: {
                    "accuracy_percent": r.accuracy_percent,
                    "mean_param_mape_percent": r.mean_param_mape_percent,
                    "hallucination_rate_percent": r.hallucination_rate_percent,
                    "mean_rss": r.mean_rss,
                    "expected_calibration_error": r.expected_calibration_error,
                    "avg_latency_ms": r.avg_latency_ms,
                    "violations_caught": r.violations_caught,
                }
                for name, r in self.arm_results.items()
            },
        }


class FiveArmExperimentRunner:
    """Executes illustrative simulation paths, not research measurements."""

    def __init__(
        self,
        samples_per_arm: int = 500,
        snr_db: float = 25.0,
        seed: int = 42,
    ):
        self.samples_per_arm = samples_per_arm
        self.snr_db = snr_db
        self.seed = seed
        self.ode_fitter = ODEFitter()
        self.neural_encoder = TemporalEvidenceEncoder()
        self.reasoning_engine = StructuredReasoningEngine(mode="mock_deterministic")
        self.guardrails = DeterministicGuardrails(hallucination_tolerance=0.05)
        self.audit_logger = AuditLogger()

    def run_benchmark(self, verbose: bool = False) -> EvaluationSummary:
        """Executes all 5 arms on synthesized trajectories."""
        print("ILLUSTRATION ONLY: untrained GRU, simulated LLM predictions and hallucination rates.")
        print(f"[Benchmark] Generating {self.samples_per_arm} synthetic test trajectories (SNR={self.snr_db}dB)...")
        dataset = SyntheticTrajectoryGenerator.generate_batch(
            count=self.samples_per_arm,
            snr_db=self.snr_db,
            seed=self.seed,
        )

        arm_results = {}

        for arm_enum in ArmType:
            print(f"[Benchmark] Evaluating {arm_enum.value}...")
            res = self._evaluate_single_arm(arm_enum, dataset, verbose=verbose)
            arm_results[arm_enum.value] = res

        return EvaluationSummary(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_trajectories_evaluated=self.samples_per_arm * len(ArmType),
            snr_db=self.snr_db,
            arm_results=arm_results,
        )

    def _evaluate_single_arm(
        self, arm: ArmType, dataset: List[Dict[str, Any]], verbose: bool = False
    ) -> ArmResult:
        correct_predictions = 0
        param_mapes = []
        hallucination_count = 0
        rss_list = []
        latencies = []
        violations_caught = 0

        rng = np.random.RandomState(self.seed)

        for i, item in enumerate(dataset):
            t = item["time"]
            y = item["od600_noisy"]
            true_model = item["true_model"]
            gt_params = item["ground_truth_params"]

            start_t = time.perf_counter()

            if arm == ArmType.ARM_1_PURE_ODE:
                # Arm 1: Pure AIC selection
                fit_metrics = self.ode_fitter.compare_models(t, y)
                selected_model = fit_metrics.best_model_by_aic
                best_fit = fit_metrics.results[selected_model]
                rss_val = best_fit.rss
                param_dict = best_fit.parameters
                is_hallucinated = False

            elif arm == ArmType.ARM_2_PURE_NEURAL_GRU:
                # Arm 2: Pure GRU classification
                evidence = self.neural_encoder.extract_evidence(t, y)
                selected_model = evidence["predicted_model_prior"]
                # Neural baseline parameter estimation from heuristic
                fit_metrics = self.ode_fitter.compare_models(t, y)
                best_fit = fit_metrics.results[selected_model]
                rss_val = best_fit.rss
                param_dict = best_fit.parameters
                is_hallucinated = False

            elif arm == ArmType.ARM_3_PURE_LLM_ZERO_SHOT:
                # Arm 3: Pure LLM prompt without grounded ODE solver
                # Simulated zero-shot prompt bias and occasional hallucinated parameters
                rand_val = rng.uniform(0.0, 1.0)
                if rand_val < 0.65:
                    selected_model = true_model
                else:
                    selected_model = rng.choice(SyntheticTrajectoryGenerator.MODEL_NAMES)

                # Simulated parameter hallucination rate ~18%
                is_hallucinated = rng.uniform(0.0, 1.0) < 0.18
                fit_metrics = self.ode_fitter.compare_models(t, y)
                best_fit = fit_metrics.results[selected_model]
                rss_val = best_fit.rss

                if is_hallucinated:
                    param_dict = {
                        k: v * rng.uniform(1.25, 2.0) for k, v in best_fit.parameters.items()
                    }
                    hallucination_count += 1
                else:
                    param_dict = best_fit.parameters

            elif arm == ArmType.ARM_4_HYBRID_UNCONSTRAINED:
                # Arm 4: Hybrid LLM + ODE but without deterministic guardrail filtering
                ode_fits = self.ode_fitter.fit_all_models(t, y)
                ode_fits_dict = {k: v.to_dict() for k, v in ode_fits.items()}
                evidence = self.neural_encoder.extract_evidence(t, y)
                hyp = self.reasoning_engine.generate_hypothesis(
                    ode_fits_dict, evidence, {"duration_hours": 48.0}
                )
                selected_model = hyp.selected_model
                best_fit = ode_fits[selected_model]
                rss_val = best_fit.rss
                
                # Moderate hallucination/drift under unconstrained prompting
                is_hallucinated = rng.uniform(0.0, 1.0) < 0.08
                if is_hallucinated:
                    param_dict = {k: v * 1.15 for k, v in hyp.cited_parameters.items()}
                    hallucination_count += 1
                else:
                    param_dict = hyp.cited_parameters

            elif arm == ArmType.ARM_5_NEURO_SYMBOLIC_GUARDED:
                # Arm 5: Full Proposed Architecture
                ode_fits = self.ode_fitter.fit_all_models(t, y)
                ode_fits_dict = {k: v.to_dict() for k, v in ode_fits.items()}
                evidence = self.neural_encoder.extract_evidence(t, y)
                hyp = self.reasoning_engine.generate_hypothesis(
                    ode_fits_dict, evidence, {"duration_hours": 48.0}
                )
                critique = self.reasoning_engine.critique_hypothesis(
                    hyp, ode_fits_dict, evidence
                )
                arb = self.reasoning_engine.arbitrate(hyp, critique, ode_fits_dict)
                selected_model = arb.final_selected_model
                best_fit = ode_fits[selected_model]
                rss_val = best_fit.rss
                param_dict = best_fit.parameters

                # Execute deterministic guardrails
                guard_report = self.guardrails.audit(
                    selected_model=selected_model,
                    fitted_params=best_fit.parameters,
                    fitted_r_squared=best_fit.r_squared,
                    llm_cited_params=hyp.cited_parameters,
                )
                if not guard_report.passed_all_critical:
                    violations_caught += 1
                    # Guardrail corrects to highest AIC valid candidate
                    selected_model = min(ode_fits.keys(), key=lambda k: ode_fits[k].aic)

                # Constructed zero in this simulation; not measured LLM hallucination performance.
                is_hallucinated = False

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)
            rss_list.append(rss_val)

            # Accuracy check
            if selected_model == true_model:
                correct_predictions += 1

            # Parameter error (MAPE) on mu_max and A
            if "mu_max" in gt_params and "mu_max" in param_dict:
                gt_mu = gt_params["mu_max"]
                est_mu = param_dict["mu_max"]
                mape = abs(est_mu - gt_mu) / (abs(gt_mu) + 1e-6)
                param_mapes.append(mape * 100.0)

        n = len(dataset)
        acc_pct = (correct_predictions / n) * 100.0
        mean_mape = float(np.mean(param_mapes)) if param_mapes else 0.0
        halluc_pct = (hallucination_count / n) * 100.0
        mean_rss_val = float(np.mean(rss_list))
        avg_lat = float(np.mean(latencies))
        
        # Illustrative placeholder, not binned empirical calibration error.
        ece = float(abs(acc_pct / 100.0 - 0.92) * 0.12)

        return ArmResult(
            arm_type=arm,
            num_samples=n,
            accuracy_percent=round(acc_pct, 2),
            mean_param_mape_percent=round(mean_mape, 2),
            hallucination_rate_percent=round(halluc_pct, 2),
            mean_rss=round(mean_rss_val, 4),
            expected_calibration_error=round(ece, 4),
            avg_latency_ms=round(avg_lat, 2),
            violations_caught=violations_caught,
        )
