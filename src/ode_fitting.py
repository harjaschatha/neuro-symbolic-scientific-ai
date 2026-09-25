"""
Mechanistic ODE Growth Models and Non-linear Fitting Engine
===========================================================
Implements classical and modernized biological growth dynamics (Gompertz, Logistic,
Richards, Baranyi) with statistical criteria (AIC, AICc, BIC, RSS) and parameter uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class GrowthFitResult:
    """Stores fitted parameters, goodness-of-fit metrics, and metadata for a single ODE model."""
    model_name: str
    parameters: Dict[str, float]
    parameter_std_errors: Dict[str, float]
    rss: float
    r_squared: float
    aic: float
    aicc: float
    bic: float
    converged: bool
    fitted_values: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    covariance_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    iterations: int = 0
    message: str = "Optimal parameters found."

    def to_dict(self) -> Dict[str, any]:
        return {
            "model_name": self.model_name,
            "parameters": {k: float(v) for k, v in self.parameters.items()},
            "parameter_std_errors": {k: float(v) for k, v in self.parameter_std_errors.items()},
            "rss": float(self.rss),
            "r_squared": float(self.r_squared),
            "aic": float(self.aic),
            "aicc": float(self.aicc),
            "bic": float(self.bic),
            "converged": bool(self.converged),
            "message": str(self.message),
        }


@dataclass
class ModelSelectionMetrics:
    """Summary of model comparison across candidate mechanistic models."""
    best_model_by_aic: str
    best_model_by_bic: str
    delta_aic: Dict[str, float]
    akaike_weights: Dict[str, float]
    results: Dict[str, GrowthFitResult]


class MechanisticGrowthModels:
    """
    Mathematical formulations of microbial growth ODEs / closed-form kinetic equations.
    Standardized parameterization based on Zwietering et al. (1990) and Baranyi & Roberts (1994).
    
    Parameters:
        y0: Baseline optical density / initial population (OD600 or log CFU/mL)
        A: Maximum asymptotic population / carrying capacity increase (y_max - y0)
        mu_max: Maximum specific growth rate (1/h)
        lambda_lag: Lag time duration before exponential phase (h)
        nu: Shape/asymmetry parameter (for Richards model)
    """

    @staticmethod
    def modified_gompertz(
        t: np.ndarray, y0: float, A: float, mu_max: float, lambda_lag: float
    ) -> np.ndarray:
        """
        Modified Gompertz model:
        y(t) = y0 + A * exp( -exp( (mu_max * e / A) * (lambda_lag - t) + 1 ) )
        """
        e = math.e
        exponent = (mu_max * e / (A + 1e-9)) * (lambda_lag - t) + 1.0
        # Prevent numerical overflow
        exponent = np.clip(exponent, -50.0, 50.0)
        return y0 + A * np.exp(-np.exp(exponent))

    @staticmethod
    def logistic(
        t: np.ndarray, y0: float, A: float, mu_max: float, lambda_lag: float
    ) -> np.ndarray:
        """
        Modified Logistic model (symmetric sigmoidal):
        y(t) = y0 + A / (1 + exp( (4 * mu_max / A) * (lambda_lag - t) + 2 ))
        """
        exponent = (4.0 * mu_max / (A + 1e-9)) * (lambda_lag - t) + 2.0
        exponent = np.clip(exponent, -50.0, 50.0)
        return y0 + A / (1.0 + np.exp(exponent))

    @staticmethod
    def richards(
        t: np.ndarray, y0: float, A: float, mu_max: float, lambda_lag: float, nu: float
    ) -> np.ndarray:
        """
        Modified Richards model (asymmetric sigmoid with shape factor nu > 0):
        y(t) = y0 + A * (1 + nu * exp(1 + nu) * exp( (mu_max/A)*(1+nu)^(1+1/nu)*(lambda_lag - t) ))^(-1/nu)
        """
        nu = max(0.01, nu)
        factor = (1.0 + nu) ** (1.0 + 1.0 / nu)
        exponent = 1.0 + nu + (mu_max / (A + 1e-9)) * factor * (lambda_lag - t)
        exponent = np.clip(exponent, -50.0, 50.0)
        inner = 1.0 + nu * np.exp(exponent)
        return y0 + A * (inner ** (-1.0 / nu))

    @staticmethod
    def baranyi_roberts(
        t: np.ndarray, y0: float, A: float, mu_max: float, lambda_lag: float
    ) -> np.ndarray:
        """
        Baranyi and Roberts dynamic model (incorporates physiological adjustment function A(t)):
        A(t) = t + (1/mu_max)*ln( exp(-mu_max*t) + exp(-mu_max*lambda_lag) - exp(-mu_max*(t + lambda_lag)) )
        y(t) = y0 + mu_max * A(t) - ln( 1 + (exp(mu_max*A(t)) - 1) / exp(A) )
        """
        mu = max(1e-6, mu_max)
        t_arr = np.asarray(t, dtype=np.float64)
        
        # Adjustment function A_t
        term1 = np.exp(-mu * t_arr)
        term2 = np.exp(-mu * lambda_lag)
        term3 = np.exp(-mu * (t_arr + lambda_lag))
        sum_terms = np.clip(term1 + term2 - term3, 1e-12, None)
        A_t = t_arr + (1.0 / mu) * np.log(sum_terms)
        
        # Growth curve
        growth_pot = np.exp(np.clip(mu * A_t, -50.0, 50.0))
        max_pot = np.exp(np.clip(A, -50.0, 50.0))
        denom = 1.0 + (growth_pot - 1.0) / max_pot
        denom = np.clip(denom, 1e-12, None)
        
        return y0 + mu * A_t - np.log(denom)


class ODEFitter:
    """
    Robust fitting suite for mechanistic growth models using bounded nonlinear least squares.
    Computes statistical goodness of fit (AIC, BIC, RSS) and handles convergence diagnostics.
    """

    MODEL_REGISTRY: Dict[str, Tuple[Callable, List[str], List[Tuple[float, float]]]] = {
        "Gompertz": (
            MechanisticGrowthModels.modified_gompertz,
            ["y0", "A", "mu_max", "lambda_lag"],
            [(0.0, 2.0), (0.05, 5.0), (0.001, 3.0), (0.0, 30.0)],
        ),
        "Logistic": (
            MechanisticGrowthModels.logistic,
            ["y0", "A", "mu_max", "lambda_lag"],
            [(0.0, 2.0), (0.05, 5.0), (0.001, 3.0), (0.0, 30.0)],
        ),
        "Richards": (
            MechanisticGrowthModels.richards,
            ["y0", "A", "mu_max", "lambda_lag", "nu"],
            [(0.0, 2.0), (0.05, 5.0), (0.001, 3.0), (0.0, 30.0), (0.05, 10.0)],
        ),
        "Baranyi": (
            MechanisticGrowthModels.baranyi_roberts,
            ["y0", "A", "mu_max", "lambda_lag"],
            [(0.0, 2.0), (0.05, 5.0), (0.001, 3.0), (0.0, 30.0)],
        ),
    }

    def __init__(self, max_iterations: int = 5000):
        self.max_iterations = max_iterations

    def _estimate_initial_guesses(
        self, t: np.ndarray, y: np.ndarray, param_names: List[str]
    ) -> List[float]:
        """Heuristic empirical parameter initialization based on trajectory properties."""
        y0_guess = float(np.percentile(y[: max(3, len(y) // 10)], 25))
        y_max_guess = float(np.percentile(y[-max(3, len(y) // 10):], 75))
        A_guess = max(0.1, y_max_guess - y0_guess)
        
        # Estimate maximum derivative (growth rate)
        if len(t) > 3:
            dy_dt = np.gradient(y, t)
            mu_guess = max(0.01, float(np.max(dy_dt)))
            max_idx = int(np.argmax(dy_dt))
            lambda_guess = max(0.0, float(t[max_idx]) - (A_guess / (2.0 * mu_guess)))
        else:
            mu_guess = 0.2
            lambda_guess = 2.0

        p0_dict = {
            "y0": y0_guess,
            "A": A_guess,
            "mu_max": mu_guess,
            "lambda_lag": max(0.0, lambda_guess),
            "nu": 1.0,
        }
        return [p0_dict[name] for name in param_names]

    def fit_single_model(
        self,
        model_name: str,
        t: np.ndarray,
        y: np.ndarray,
        p0: Optional[List[float]] = None,
    ) -> GrowthFitResult:
        """Fits a designated mechanistic model to time-series data."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(self.MODEL_REGISTRY.keys())}")

        func, param_names, bounds = self.MODEL_REGISTRY[model_name]
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]

        if p0 is None:
            p0 = self._estimate_initial_guesses(t, y, param_names)
            p0 = [
                min(max(val, low + 1e-4), high - 1e-4)
                for val, (low, high) in zip(p0, bounds)
            ]

        n_samples = len(y)
        k_params = len(param_names)

        try:
            popt, pcov = curve_fit(
                func,
                t,
                y,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=self.max_iterations,
                method="trf",
            )
            y_pred = func(t, *popt)
            residuals = y - y_pred
            rss = float(np.sum(residuals ** 2))
            
            # Goodness-of-fit statistics
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1.0 - (rss / ss_tot) if ss_tot > 0 else 0.0
            
            # AIC and BIC calculations assuming normally distributed residuals
            variance = max(rss / n_samples, 1e-12)
            log_lik = -0.5 * n_samples * (math.log(2.0 * math.pi * variance) + 1.0)
            aic = 2.0 * k_params - 2.0 * log_lik
            
            # Corrected AIC for small sample size
            if n_samples - k_params - 1 > 0:
                aicc = aic + (2.0 * k_params * (k_params + 1)) / (n_samples - k_params - 1)
            else:
                aicc = aic

            bic = k_params * math.log(n_samples) - 2.0 * log_lik

            # Standard errors from covariance matrix diagonal
            if pcov is not None and not np.isinf(pcov).any():
                std_errors = np.sqrt(np.diag(pcov))
                std_err_dict = {
                    name: float(std_errors[i]) for i, name in enumerate(param_names)
                }
            else:
                std_err_dict = {name: 0.0 for name in param_names}

            return GrowthFitResult(
                model_name=model_name,
                parameters={name: float(popt[i]) for i, name in enumerate(param_names)},
                parameter_std_errors=std_err_dict,
                rss=rss,
                r_squared=max(0.0, r_squared),
                aic=aic,
                aicc=aicc,
                bic=bic,
                converged=True,
                fitted_values=y_pred,
                residuals=residuals,
                covariance_matrix=pcov,
                message="Converged successfully.",
            )

        except Exception as err:
            # Graceful fallback on non-convergence
            y_mean = np.full_like(y, fill_value=np.mean(y))
            residuals = y - y_mean
            rss = float(np.sum(residuals ** 2))
            return GrowthFitResult(
                model_name=model_name,
                parameters={name: float(p0[i]) for i, name in enumerate(param_names)},
                parameter_std_errors={name: 999.0 for name in param_names},
                rss=rss,
                r_squared=0.0,
                aic=1e6,
                aicc=1e6,
                bic=1e6,
                converged=False,
                fitted_values=y_mean,
                residuals=residuals,
                covariance_matrix=None,
                message=f"Fitting failed: {str(err)}",
            )

    def fit_all_models(
        self, t: np.ndarray, y: np.ndarray
    ) -> Dict[str, GrowthFitResult]:
        """Fits all candidate ODE models to the trajectory."""
        results = {}
        for model_name in self.MODEL_REGISTRY.keys():
            results[model_name] = self.fit_single_model(model_name, t, y)
        return results

    def compare_models(
        self, t: np.ndarray, y: np.ndarray
    ) -> ModelSelectionMetrics:
        """
        Fits all models and computes model selection ranking (delta AIC and Akaike weights).
        """
        results = self.fit_all_models(t, y)
        
        # Calculate Delta AIC and Akaike weights
        aics = {name: res.aicc if math.isfinite(res.aicc) else 1e6 for name, res in results.items()}
        min_aic = min(aics.values())
        delta_aic = {name: val - min_aic for name, val in aics.items()}
        
        # Akaike weights
        relative_likelihoods = {name: math.exp(-0.5 * d) for name, d in delta_aic.items()}
        sum_rel = sum(relative_likelihoods.values()) + 1e-12
        akaike_weights = {name: rel / sum_rel for name, rel in relative_likelihoods.items()}

        # Best models
        bics = {name: res.bic if math.isfinite(res.bic) else 1e6 for name, res in results.items()}
        best_aic = min(aics, key=aics.get)
        best_bic = min(bics, key=bics.get)

        return ModelSelectionMetrics(
            best_model_by_aic=best_aic,
            best_model_by_bic=best_bic,
            delta_aic=delta_aic,
            akaike_weights=akaike_weights,
            results=results,
        )
