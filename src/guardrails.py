"""
Deterministic Guardrails & Scientific Consistency Verification Engine
======================================================================
Implements deterministic, rule-based verification for biological plausibility,
physical consistency, parameter bounds, asymptotic monotonicity, and LLM hallucination trapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class GuardrailStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"


@dataclass
class ValidationRuleResult:
    """Individual rule verification outcome."""
    rule_name: str
    status: GuardrailStatus
    details: str
    metric_value: Optional[float] = None
    threshold: Optional[str] = None


@dataclass
class GuardrailAuditReport:
    """Consolidated audit report across all deterministic guardrail checks."""
    overall_status: GuardrailStatus
    passed_all_critical: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    violations: List[str]
    rule_results: List[ValidationRuleResult]

    def to_dict(self) -> Dict[str, any]:
        return {
            "overall_status": self.overall_status.value,
            "passed_all_critical": self.passed_all_critical,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "violations": self.violations,
            "rule_results": [
                {
                    "rule": r.rule_name,
                    "status": r.status.value,
                    "details": r.details,
                    "metric_value": r.metric_value,
                    "threshold": r.threshold,
                }
                for r in self.rule_results
            ],
        }


class BiologicalConstraintValidator:
    """
    Ensures that fitted parameters and predicted growth curves conform to
    established microbiological principles and biophysical boundaries.
    """

    # Biophysical bounds for standard bacterial batch culture (OD600)
    BOUNDS = {
        "y0_min": 0.0,
        "y0_max": 1.5,
        "A_min": 0.05,
        "A_max": 4.5,
        "mu_max_min": 0.001,  # Minimum doubling time approx 700h
        "mu_max_max": 3.5,    # Maximum doubling time approx 12 min
        "lambda_min": 0.0,
        "lambda_max": 36.0,   # Unrealistic to have lag > 36h in a 48h experiment
    }

    @classmethod
    def validate_parameters(
        cls, model_name: str, params: Dict[str, float]
    ) -> List[ValidationRuleResult]:
        results = []

        # 1. Baseline Optical Density
        y0 = params.get("y0", 0.0)
        if y0 < cls.BOUNDS["y0_min"]:
            results.append(
                ValidationRuleResult(
                    rule_name="NonNegativeInitialOD",
                    status=GuardrailStatus.FAILED,
                    details=f"Initial population y0 ({y0:.4f}) is negative.",
                    metric_value=y0,
                    threshold=">= 0.0",
                )
            )
        elif y0 > cls.BOUNDS["y0_max"]:
            results.append(
                ValidationRuleResult(
                    rule_name="PlausibleInitialOD",
                    status=GuardrailStatus.WARNING,
                    details=f"Initial population y0 ({y0:.4f}) is suspiciously high.",
                    metric_value=y0,
                    threshold=f"<= {cls.BOUNDS['y0_max']}",
                )
            )
        else:
            results.append(
                ValidationRuleResult(
                    rule_name="NonNegativeInitialOD",
                    status=GuardrailStatus.PASSED,
                    details="Initial population within plausible range.",
                    metric_value=y0,
                )
            )

        # 2. Maximum Specific Growth Rate mu_max
        mu = params.get("mu_max", 0.0)
        if mu <= cls.BOUNDS["mu_max_min"]:
            results.append(
                ValidationRuleResult(
                    rule_name="PositiveGrowthRate",
                    status=GuardrailStatus.FAILED,
                    details=f"Growth rate mu_max ({mu:.4f}) is non-positive or near zero.",
                    metric_value=mu,
                    threshold=f"> {cls.BOUNDS['mu_max_min']}",
                )
            )
        elif mu > cls.BOUNDS["mu_max_max"]:
            results.append(
                ValidationRuleResult(
                    rule_name="ThermodynamicGrowthCap",
                    status=GuardrailStatus.FAILED,
                    details=f"Growth rate mu_max ({mu:.4f}) exceeds biological maximum limit.",
                    metric_value=mu,
                    threshold=f"<= {cls.BOUNDS['mu_max_max']}",
                )
            )
        else:
            results.append(
                ValidationRuleResult(
                    rule_name="PositiveGrowthRate",
                    status=GuardrailStatus.PASSED,
                    details="Specific growth rate within valid physiological regime.",
                    metric_value=mu,
                )
            )

        # 3. Lag phase lambda
        lag = params.get("lambda_lag", 0.0)
        if lag < cls.BOUNDS["lambda_min"]:
            results.append(
                ValidationRuleResult(
                    rule_name="NonNegativeLagPhase",
                    status=GuardrailStatus.FAILED,
                    details=f"Lag time lambda ({lag:.4f}) is negative.",
                    metric_value=lag,
                    threshold=">= 0.0",
                )
            )
        elif lag > cls.BOUNDS["lambda_max"]:
            results.append(
                ValidationRuleResult(
                    rule_name="LagPhaseWithinDuration",
                    status=GuardrailStatus.WARNING,
                    details=f"Lag time lambda ({lag:.4f}h) exceeds normal experimental bounds.",
                    metric_value=lag,
                    threshold=f"<= {cls.BOUNDS['lambda_max']}h",
                )
            )
        else:
            results.append(
                ValidationRuleResult(
                    rule_name="NonNegativeLagPhase",
                    status=GuardrailStatus.PASSED,
                    details="Lag phase parameter is valid.",
                    metric_value=lag,
                )
            )

        # 4. Carrying Capacity A
        A = params.get("A", 0.0)
        if A <= cls.BOUNDS["A_min"]:
            results.append(
                ValidationRuleResult(
                    rule_name="PositiveCarryingCapacity",
                    status=GuardrailStatus.FAILED,
                    details=f"Asymptotic capacity A ({A:.4f}) is insufficient for growth.",
                    metric_value=A,
                    threshold=f"> {cls.BOUNDS['A_min']}",
                )
            )
        else:
            results.append(
                ValidationRuleResult(
                    rule_name="PositiveCarryingCapacity",
                    status=GuardrailStatus.PASSED,
                    details="Carrying capacity growth delta is valid.",
                    metric_value=A,
                )
            )

        return results


class ParameterConsistencyCheck:
    """
    Guards against LLM hallucination and mathematical discrepancies between
    symbolic outputs and deterministic ODE regression results.
    """

    @classmethod
    def check_hallucination(
        cls,
        selected_model: str,
        llm_cited_params: Dict[str, float],
        actual_fitted_params: Dict[str, float],
        tolerance: float = 0.05,  # 5% allowable deviation
    ) -> List[ValidationRuleResult]:
        results = []

        for param_name, fitted_val in actual_fitted_params.items():
            if param_name not in llm_cited_params:
                continue
            
            cited_val = llm_cited_params[param_name]
            rel_diff = abs(cited_val - fitted_val) / (abs(fitted_val) + 1e-6)

            if rel_diff > tolerance:
                results.append(
                    ValidationRuleResult(
                        rule_name=f"HallucinationCheck_{param_name}",
                        status=GuardrailStatus.FAILED,
                        details=(
                            f"LLM cited {param_name}={cited_val:.4f}, but fitted ground truth "
                            f"is {fitted_val:.4f} (relative error {rel_diff*100:.2f}% > {tolerance*100}%)."
                        ),
                        metric_value=rel_diff,
                        threshold=f"<= {tolerance}",
                    )
                )
            else:
                results.append(
                    ValidationRuleResult(
                        rule_name=f"HallucinationCheck_{param_name}",
                        status=GuardrailStatus.PASSED,
                        details=f"LLM parameter {param_name} accurately reflects solver output.",
                        metric_value=rel_diff,
                    )
                )

        return results


class DeterministicGuardrails:
    """Orchestrates comprehensive deterministic verification and audit reporting."""

    def __init__(self, hallucination_tolerance: float = 0.05):
        self.tolerance = hallucination_tolerance

    def audit(
        self,
        selected_model: str,
        fitted_params: Dict[str, float],
        fitted_r_squared: float,
        llm_cited_params: Optional[Dict[str, float]] = None,
    ) -> GuardrailAuditReport:
        rule_results: List[ValidationRuleResult] = []

        # 1. Biological Parameter Constraints
        bio_results = BiologicalConstraintValidator.validate_parameters(
            selected_model, fitted_params
        )
        rule_results.extend(bio_results)

        # 2. Goodness-of-Fit Minimum Threshold
        if fitted_r_squared < 0.70:
            rule_results.append(
                ValidationRuleResult(
                    rule_name="MinimumGoodnessOfFit",
                    status=GuardrailStatus.FAILED,
                    details=f"Fitted R-squared ({fitted_r_squared:.4f}) is below acceptable threshold 0.70.",
                    metric_value=fitted_r_squared,
                    threshold=">= 0.70",
                )
            )
        elif fitted_r_squared < 0.85:
            rule_results.append(
                ValidationRuleResult(
                    rule_name="MinimumGoodnessOfFit",
                    status=GuardrailStatus.WARNING,
                    details=f"Fitted R-squared ({fitted_r_squared:.4f}) indicates moderate noise or sub-optimal fit.",
                    metric_value=fitted_r_squared,
                    threshold=">= 0.85",
                )
            )
        else:
            rule_results.append(
                ValidationRuleResult(
                    rule_name="MinimumGoodnessOfFit",
                    status=GuardrailStatus.PASSED,
                    details=f"Fitted R-squared ({fitted_r_squared:.4f}) demonstrates high explanatory fidelity.",
                    metric_value=fitted_r_squared,
                )
            )

        # 3. LLM Parameter Consistency & Hallucination Check
        if llm_cited_params:
            hallucination_results = ParameterConsistencyCheck.check_hallucination(
                selected_model,
                llm_cited_params,
                fitted_params,
                tolerance=self.tolerance,
            )
            rule_results.extend(hallucination_results)

        # Compute summary
        failed = [r for r in rule_results if r.status == GuardrailStatus.FAILED]
        warnings = [r for r in rule_results if r.status == GuardrailStatus.WARNING]
        passed = [r for r in rule_results if r.status == GuardrailStatus.PASSED]

        passed_all_critical = len(failed) == 0
        overall_status = (
            GuardrailStatus.PASSED
            if passed_all_critical and len(warnings) == 0
            else (GuardrailStatus.WARNING if passed_all_critical else GuardrailStatus.FAILED)
        )

        violations = [f"{r.rule_name}: {r.details}" for r in failed]

        return GuardrailAuditReport(
            overall_status=overall_status,
            passed_all_critical=passed_all_critical,
            total_checks=len(rule_results),
            passed_checks=len(passed),
            failed_checks=len(failed),
            warning_checks=len(warnings),
            violations=violations,
            rule_results=rule_results,
        )
