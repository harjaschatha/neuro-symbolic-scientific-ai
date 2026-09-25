# Methodology and scope

## Evaluated research system

The research results concern biological mechanism selection from synthetic
multivariate bioprocess trajectories. The earlier library fits baseline additive
dual-substrate Monod growth, maintenance, and inhibition. The later expanded
library makes additional labels directly selectable, including switching,
switching-inhibition, product inhibition, oxygen limitation,
maintenance-inhibition, and yield drift.

The source workflow combines temporal evidence, biological diagnostics,
structured hypotheses, mechanistic fitting, ranking, and deterministic selection
rules. The earlier report artifact records `seqmlp` residuals in all 500 cases
and frozen-classifier hypotheses in 464 cases. It is not an LLM-only evaluation.

Exact biological labels and projected surrogate labels answer different
questions. In the earlier study, switching can be counted as correct when
represented by maintenance. In the paired Stage E study the two arms have
different saved projections. Common-label exact and fixed-family comparisons
are therefore reported alongside saved accuracy. See
[metric definitions](experiment_protocol.md) and the
[calculation script](../scripts/export_verified_results.py).

## Separate curve-fitting illustration

`src/ode_fitting.py` implements analytic growth curves and nonlinear fitting;
these are not the Monod mechanism library used in the reported research.
`src/gru_model.py` constructs an untrained GRU by default. Its output probabilities
and attention locations are not validated biological evidence.
`src/llm_reasoning.py` produces mock deterministic hypotheses and critiques.
Pydantic validation checks payload structure, not scientific correctness.

The simulated five-arm runner uses artificial prediction and drift rules. Its
parameter error covers only `mu_max`; its hallucination counts and calibration
approximation are not empirical LLM evaluation metrics. It should only be used
for illustration of software paths.

The demo's curve parameter named `mu_max` should not automatically be interpreted
as a specific growth rate in inverse hours: for the implemented modified
Gompertz and Logistic curves it controls the maximum slope of the supplied
response variable. Physical interpretation depends on the data representation.
No mechanistic equivalence between the demo and evaluated research is asserted.
