"""
Audit and Provenance Logging Layer
==================================
Provides cryptographic traceability, reproducibility logging, and SHA-256
hash verification across all stages of the neuro-symbolic decision pipeline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def compute_sha256(data: Any) -> str:
    """Computes deterministic SHA-256 hash from JSON-serializable Python object."""
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ProvenanceRecord:
    """Cryptographically anchored provenance record for an individual model selection run."""
    run_id: str
    timestamp_utc: str
    input_data_hash: str
    ode_fits_hash: str
    neural_evidence_hash: str
    llm_reasoning_hash: str
    guardrail_audit_hash: str
    final_decision_hash: str
    selected_model: str
    guardrail_verdict: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Manages audit records and provides integrity verification across research runs."""

    def __init__(self, log_filepath: Optional[str] = None):
        self.log_filepath = log_filepath
        self.records: List[ProvenanceRecord] = []

    def create_record(
        self,
        run_id: str,
        raw_trajectory: Dict[str, Any],
        ode_fits: Dict[str, Any],
        neural_evidence: Dict[str, Any],
        llm_reasoning: Dict[str, Any],
        guardrail_report: Dict[str, Any],
        selected_model: str,
        execution_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        now_str = datetime.now(timezone.utc).isoformat()

        input_hash = compute_sha256({
            "t": [round(float(x), 4) for x in raw_trajectory.get("time", [])],
            "y": [round(float(x), 4) for x in raw_trajectory.get("od600_noisy", [])],
        })
        ode_hash = compute_sha256(ode_fits)
        neural_hash = compute_sha256(neural_evidence)
        llm_hash = compute_sha256(llm_reasoning)
        guardrail_hash = compute_sha256(guardrail_report)

        decision_payload = {
            "selected_model": selected_model,
            "input_hash": input_hash,
            "ode_hash": ode_hash,
            "neural_hash": neural_hash,
            "llm_hash": llm_hash,
            "guardrail_hash": guardrail_hash,
        }
        final_hash = compute_sha256(decision_payload)

        record = ProvenanceRecord(
            run_id=run_id,
            timestamp_utc=now_str,
            input_data_hash=input_hash,
            ode_fits_hash=ode_hash,
            neural_evidence_hash=neural_hash,
            llm_reasoning_hash=llm_hash,
            guardrail_audit_hash=guardrail_hash,
            final_decision_hash=final_hash,
            selected_model=selected_model,
            guardrail_verdict=guardrail_report.get("overall_status", "UNKNOWN"),
            execution_time_ms=round(execution_time_ms, 2),
            metadata=metadata or {},
        )

        self.records.append(record)

        if self.log_filepath:
            try:
                with open(self.log_filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except IOError as err:
                print(f"[AuditLogger] Warning: Failed to append to audit log: {err}")

        return record

    def verify_record_integrity(
        self,
        record: ProvenanceRecord,
        raw_trajectory: Dict[str, Any],
        ode_fits: Dict[str, Any],
        neural_evidence: Dict[str, Any],
        llm_reasoning: Dict[str, Any],
        guardrail_report: Dict[str, Any],
    ) -> bool:
        """Verifies whether the recorded cryptographic hashes match recomputed values."""
        input_hash = compute_sha256({
            "t": [round(float(x), 4) for x in raw_trajectory.get("time", [])],
            "y": [round(float(x), 4) for x in raw_trajectory.get("od600_noisy", [])],
        })
        ode_hash = compute_sha256(ode_fits)
        neural_hash = compute_sha256(neural_evidence)
        llm_hash = compute_sha256(llm_reasoning)
        guardrail_hash = compute_sha256(guardrail_report)

        return (
            record.input_data_hash == input_hash
            and record.ode_fits_hash == ode_hash
            and record.neural_evidence_hash == neural_hash
            and record.llm_reasoning_hash == llm_hash
            and record.guardrail_audit_hash == guardrail_hash
        )
