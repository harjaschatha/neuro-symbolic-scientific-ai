# Historical illustrative outputs — not research results

All files in this directory belong to the separate curve-fitting demo:

- `synthetic-example.json`: generated curve, fitted models, untrained neural
  output, mock reasoning, and guardrail checks.
- `synthetic_benchmark_sample.json`: 12 generated curves across four models and
  three noise settings; this is not the research evaluation cohort.
- `audit_trail.jsonl`: historical demo records. Their hashes do not make the
  file immutable or establish research provenance.

These artifacts are preserved as historical demo outputs. Their neural scores,
reasoning text, and timings must not be cited as trained-model or live-LLM
performance. Running `run_demo.py` updates the case study and appends a demo log.
Actual reported evidence is in [`results/verified`](../results/verified/README.md).
