# Experimental Protocol & HPC Operations Guide

## 1. Five-Arm Experimental Benchmark Design

To systematically evaluate the contributions of mechanistic fitting, neural temporal priors, LLM reasoning, and deterministic guardrails, we establish a standardized **5-Arm Comparative Benchmark** over $N = 500$ controlled synthetic microbial growth trajectories per arm ($N_{total} = 2,500$).

### Benchmark Arms:
1. **Arm 1: Pure Mechanistic ODE Baseline**
   - Fits Gompertz, Logistic, Richards, and Baranyi models via non-linear least squares.
   - Selects the winning model solely on minimum Akaike Information Criterion ($\text{AICc}$).
2. **Arm 2: Pure Neural GRU Baseline**
   - End-to-end classification using the Bidirectional GRU with attention pooling without explicit ODE residual minimization.
3. **Arm 3: Pure LLM Zero-Shot Baseline**
   - Zero-shot prompt evaluation presenting raw time-series measurements directly to an LLM without grounded ODE optimization.
4. **Arm 4: Hybrid Unconstrained (No Guardrails)**
   - Couples ODE fitting and GRU priors to LLM reasoning, but omits deterministic guardrails, allowing potential parameter hallucination or unvalidated model selection.
5. **Arm 5: Full Guarded Neuro-Symbolic Framework (Proposed)**
   - Complete pipeline: Mechanistic ODE Suite + GRU Temporal Attention + Structured Reasoning + Multi-Agent Critique + Arbitration + Deterministic Biophysical Guardrails + SHA-256 Audit Trail.

---

## 2. Evaluation Metrics

| Metric | Mathematical Definition | Target Direction |
|---|---|---|
| **Model Selection Accuracy** | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{M}_i = M_i^*)$ | Higher ($\uparrow$) |
| **Mean Parameter Error (MAPE)** | $\frac{100\%}{P} \sum_{p=1}^P \left\| \frac{\hat{\theta}_p - \theta_p^*}{\theta_p^*} \right\|$ | Lower ($\downarrow$) |
| **Hallucination / Inadmissible Rate** | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Guardrail Violation or Hallucination})$ | Lower ($\downarrow$, target: 0%) |
| **Residual Sum of Squares (RSS)** | $\sum_{t} (y_t - \hat{y}_t)^2$ | Lower ($\downarrow$) |
| **Expected Calibration Error (ECE)** | $\sum_{m=1}^M \frac{\|B_m\|}{N} \|\text{acc}(B_m) - \text{conf}(B_m)\|$ | Lower ($\downarrow$) |
| **Decision Latency** | Time per trajectory execution ($\text{ms}$) | Lower ($\downarrow$) |

---

## 3. High-Performance Computing (HPC) & Slurm Architecture

### 3.1 Slurm Batch Job Template (`submit_benchmark.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=neuro_symbolic_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# 1. Activate Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate axolotl-m4

# 2. Launch Local vLLM Inference Server in Background
export VLLM_PORT=8000
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 4096 \
    --port $VLLM_PORT &

VLLM_PID=$!
echo "vLLM server launched with PID $VLLM_PID. Awaiting readiness..."

# 3. Health Check Polling
until curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null; do
    sleep 3
done
echo "vLLM server ready. Executing 5-Arm Experiment..."

# 4. Run Benchmark Suite
python -m src.evaluation --samples-per-arm 500 --snr-db 25.0 --output-dir results/

# 5. Cleanup Server
kill -9 $VLLM_PID
echo "Experiment completed successfully."
```

---

## 4. Research Operations & Reproducibility Protocol

1. **Deterministic Random Seeds:**
   - Synthetic trajectory generation seeds are strictly derived from index offsets ($seed = 42 + i$).
2. **Temperature Management:**
   - LLM generation temperature is set to $T = 0.0$ for deterministic greedy decoding.
3. **Data Isolation & Governance:**
   - All benchmarks are executed strictly against controlled synthetic kinetics.
   - Non-public research assets are physically isolated and never ingested into public repositories.
