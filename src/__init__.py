"""
Neuro-Symbolic AI Framework for Scientific Model Selection
MSc Research Project — Safe Portfolio & Synthetic Benchmark Package
"""

from src.ode_fitting import (
    MechanisticGrowthModels,
    ODEFitter,
    GrowthFitResult,
    ModelSelectionMetrics,
)
from src.gru_model import (
    TemporalEvidenceGRU,
    TemporalEvidenceEncoder,
    SyntheticTrajectoryGenerator,
    TrajectoryDataset,
)
from src.llm_reasoning import (
    ModelHypothesis,
    CritiqueReport,
    ArbitrationDecision,
    StructuredReasoningEngine,
)
from src.guardrails import (
    DeterministicGuardrails,
    GuardrailAuditReport,
    BiologicalConstraintValidator,
    ParameterConsistencyCheck,
)
from src.audit_logger import (
    AuditLogger,
    ProvenanceRecord,
)
from src.evaluation import (
    FiveArmExperimentRunner,
    ArmType,
    EvaluationSummary,
)

__version__ = "1.0.0"
__all__ = [
    "MechanisticGrowthModels",
    "ODEFitter",
    "GrowthFitResult",
    "ModelSelectionMetrics",
    "TemporalEvidenceGRU",
    "TemporalEvidenceEncoder",
    "SyntheticTrajectoryGenerator",
    "TrajectoryDataset",
    "ModelHypothesis",
    "CritiqueReport",
    "ArbitrationDecision",
    "StructuredReasoningEngine",
    "DeterministicGuardrails",
    "GuardrailAuditReport",
    "BiologicalConstraintValidator",
    "ParameterConsistencyCheck",
    "AuditLogger",
    "ProvenanceRecord",
    "FiveArmExperimentRunner",
    "ArmType",
    "EvaluationSummary",
]
