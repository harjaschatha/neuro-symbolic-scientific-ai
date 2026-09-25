#!/usr/bin/env python3
"""
Illustrative Simulation Runner — Not Research Verification
===========================================
Uses mock reasoning, an untrained GRU, and simulated LLM predictions.
Generates a multi-trajectory sample dataset in examples/synthetic_benchmark_sample.json
and executes the 5-Arm benchmark suite across diverse noise conditions.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.gru_model import SyntheticTrajectoryGenerator
from src.evaluation import FiveArmExperimentRunner


def main():
    print("[1/2] Generating benchmark sample trajectories (12 trajectories across models)...")
    sample_trajectories = []
    for model in ["Gompertz", "Logistic", "Richards", "Baranyi"]:
        for snr in [15.0, 25.0, 35.0]:
            t_data = SyntheticTrajectoryGenerator.generate_single_trajectory(
                model_name=model,
                duration=48.0,
                num_points=49,
                snr_db=snr,
                seed=hash(f"{model}_{snr}") % 10000,
            )
            sample_trajectories.append({
                "model": model,
                "snr_db": snr,
                "ground_truth_params": t_data["ground_truth_params"],
                "time": [round(float(x), 3) for x in t_data["time"]],
                "od600_noisy": [round(float(x), 4) for x in t_data["od600_noisy"]],
            })

    sample_file = PROJECT_ROOT / "examples" / "synthetic_benchmark_sample.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_trajectories, f, indent=2)
    print(f"  -> Saved {len(sample_trajectories)} samples to {sample_file.name}")

    print("\n[2/2] Running 5-Arm Experiment Runner (100 synthetic trajectories per arm illustration)...")
    runner = FiveArmExperimentRunner(samples_per_arm=100, snr_db=25.0, seed=42)
    summary = runner.run_benchmark(verbose=False)

    print("\n" + "=" * 90)
    print(f"{'ARM TYPE':48s} | {'ACCURACY':9s} | {'MAPE':7s} | {'HALLUC':8s} | {'RSS':8s} | {'LAT (ms)':8s}")
    print("-" * 90)
    for arm_name, r in summary.arm_results.items():
        print(f"{arm_name:48s} | {r.accuracy_percent:7.2f}% | {r.mean_param_mape_percent:5.2f}% | {r.hallucination_rate_percent:6.2f}% | {r.mean_rss:8.4f} | {r.avg_latency_ms:7.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
