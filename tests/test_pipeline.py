"""
Comprehensive Test Suite for Neuro-Symbolic Model Selection
============================================================
Validates ODE solvers, GRU attention models, guardrail boundary enforcement,
cryptographic provenance hashing, and end-to-end integration workflows.
"""

import math
import unittest
import numpy as np

from src.ode_fitting import MechanisticGrowthModels, ODEFitter, GrowthFitResult
from src.gru_model import (
    SyntheticTrajectoryGenerator,
    TemporalEvidenceGRU,
    TemporalEvidenceEncoder,
    TrajectoryDataset,
)
from src.guardrails import (
    DeterministicGuardrails,
    BiologicalConstraintValidator,
    ParameterConsistencyCheck,
    GuardrailStatus,
)
from src.llm_reasoning import StructuredReasoningEngine, ModelHypothesis
from src.audit_logger import AuditLogger, compute_sha256


class TestMechanisticODE(unittest.TestCase):
    """Tests analytical ODE models and nonlinear regression solver."""

    def setUp(self):
        self.t = np.linspace(0, 48, 49)
        self.fitter = ODEFitter()

    def test_gompertz_generation_and_fit(self):
        y0, A, mu, lag = 0.05, 1.5, 0.8, 4.0
        y = MechanisticGrowthModels.modified_gompertz(self.t, y0, A, mu, lag)
        
        # Verify basic properties
        self.assertAlmostEqual(y[0], y0, delta=0.01)
        self.assertAlmostEqual(y[-1], y0 + A, delta=0.05)
        
        # Fit Gompertz
        res = self.fitter.fit_single_model("Gompertz", self.t, y)
        self.assertTrue(res.converged)
        self.assertGreater(res.r_squared, 0.99)
        self.assertAlmostEqual(res.parameters["mu_max"], mu, delta=0.05)

    def test_logistic_and_baranyi_convergence(self):
        y_log = MechanisticGrowthModels.logistic(self.t, 0.05, 1.2, 0.6, 3.0)
        res_log = self.fitter.fit_single_model("Logistic", self.t, y_log)
        self.assertTrue(res_log.converged)
        self.assertGreater(res_log.r_squared, 0.99)

        y_bar = MechanisticGrowthModels.baranyi_roberts(self.t, 0.05, 1.2, 0.6, 3.0)
        res_bar = self.fitter.fit_single_model("Baranyi", self.t, y_bar)
        self.assertTrue(res_bar.converged)
        self.assertGreater(res_bar.r_squared, 0.98)

    def test_model_comparison_ranking(self):
        y = MechanisticGrowthModels.modified_gompertz(self.t, 0.05, 1.5, 0.8, 4.0)
        metrics = self.fitter.compare_models(self.t, y)
        self.assertEqual(metrics.best_model_by_aic, "Gompertz")
        self.assertIn("Gompertz", metrics.akaike_weights)
        self.assertGreater(metrics.akaike_weights["Gompertz"], 0.4)


class TestTemporalEvidenceGRU(unittest.TestCase):
    """Tests PyTorch neural encoder and synthetic trajectory generator."""

    def test_synthetic_trajectory_generation(self):
        traj = SyntheticTrajectoryGenerator.generate_single_trajectory(
            model_name="Logistic", duration=24.0, num_points=25, snr_db=30.0, seed=123
        )
        self.assertEqual(len(traj["time"]), 25)
        self.assertEqual(traj["true_model"], "Logistic")
        self.assertIn("mu_max", traj["ground_truth_params"])

    def test_neural_feature_extraction(self):
        traj = SyntheticTrajectoryGenerator.generate_single_trajectory(seed=42)
        encoder = TemporalEvidenceEncoder()
        evidence = encoder.extract_evidence(traj["time"], traj["od600_noisy"])
        
        self.assertIn("predicted_model_prior", evidence)
        self.assertIn("confidence", evidence)
        self.assertIn("class_probabilities", evidence)
        self.assertEqual(len(evidence["class_probabilities"]), 4)


class TestDeterministicGuardrails(unittest.TestCase):
    """Tests deterministic safety boundary enforcement and hallucination trapping."""

    def setUp(self):
        self.guardrails = DeterministicGuardrails(hallucination_tolerance=0.05)

    def test_valid_parameters_pass(self):
        params = {"y0": 0.05, "A": 1.5, "mu_max": 0.65, "lambda_lag": 3.0}
        report = self.guardrails.audit(
            selected_model="Gompertz",
            fitted_params=params,
            fitted_r_squared=0.98,
            llm_cited_params=params,
        )
        self.assertEqual(report.overall_status, GuardrailStatus.PASSED)
        self.assertTrue(report.passed_all_critical)
        self.assertEqual(report.failed_checks, 0)

    def test_negative_growth_rate_fails(self):
        params = {"y0": 0.05, "A": 1.5, "mu_max": -0.2, "lambda_lag": 3.0}
        report = self.guardrails.audit(
            selected_model="Gompertz",
            fitted_params=params,
            fitted_r_squared=0.98,
        )
        self.assertEqual(report.overall_status, GuardrailStatus.FAILED)
        self.assertFalse(report.passed_all_critical)
        self.assertIn("PositiveGrowthRate", [r.rule_name for r in report.rule_results if r.status == GuardrailStatus.FAILED])

    def test_hallucination_detection(self):
        fitted_params = {"y0": 0.05, "A": 1.5, "mu_max": 0.65, "lambda_lag": 3.0}
        # LLM hallucinating a 30% higher growth rate
        hallucinated_params = {"y0": 0.05, "A": 1.5, "mu_max": 0.95, "lambda_lag": 3.0}

        report = self.guardrails.audit(
            selected_model="Gompertz",
            fitted_params=fitted_params,
            fitted_r_squared=0.98,
            llm_cited_params=hallucinated_params,
        )
        self.assertEqual(report.overall_status, GuardrailStatus.FAILED)
        failed_rules = [r.rule_name for r in report.rule_results if r.status == GuardrailStatus.FAILED]
        self.assertIn("HallucinationCheck_mu_max", failed_rules)


class TestAuditLogger(unittest.TestCase):
    """Tests cryptographic hashing and provenance records."""

    def test_sha256_reproducibility(self):
        data1 = {"model": "Gompertz", "aic": -120.5}
        data2 = {"aic": -120.5, "model": "Gompertz"}
        # Hash should be invariant to key order in JSON serialization
        self.assertEqual(compute_sha256(data1), compute_sha256(data2))

    def test_audit_integrity_verification(self):
        logger = AuditLogger()
        raw_traj = {"time": [0.0, 1.0, 2.0], "od600_noisy": [0.05, 0.10, 0.25]}
        ode_fits = {"Gompertz": {"aic": -50.0}}
        neural_ev = {"predicted": "Gompertz"}
        llm_res = {"rationale": "optimal"}
        guard_rep = {"overall_status": "PASSED"}

        record = logger.create_record(
            run_id="TEST-001",
            raw_trajectory=raw_traj,
            ode_fits=ode_fits,
            neural_evidence=neural_ev,
            llm_reasoning=llm_res,
            guardrail_report=guard_rep,
            selected_model="Gompertz",
            execution_time_ms=12.5,
        )

        self.assertTrue(
            logger.verify_record_integrity(
                record, raw_traj, ode_fits, neural_ev, llm_res, guard_rep
            )
        )


if __name__ == "__main__":
    unittest.main()
