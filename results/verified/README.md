# Verified local research evidence

These artifacts were extracted from the pinned `Final_Project` files named in
`manifest.json`. The exporter reads source files without modifying them and
copies only run IDs, truth, selection, stored correctness, and source labels.
`cases.json` contains four separate 500-row result sets; it is not a 2,000-case
pooled experiment. The Stage E pair has the same 500 run IDs and truth labels.

`summary.csv` contains counts and percentages. `summary.json` additionally records
source counts, the fixed-family mapping, paired changes, and limitations.
`manifest.json` records original-file hashes and derived-artifact hashes.

The source reporting freeze was consulted at `Final_Project/REPORT_RESULTS_FREEZE.md`.
The family and null-to-baseline conventions follow
`Final_Project/scripts/44_compare_stage_e_arms.py`.

The earlier report's 89.8% remains provisional: its original HPC source has not
been reconciled with the conflicting 75.2% local ablation copy. Both are retained
separately. Stage E configuration and commit snapshots are not verified as
run-time snapshots. These are checks of saved local data, not a rerun or an
independent validation of the scientific evaluator.

Run `python scripts/export_verified_results.py --check` to verify calculations
and bundled hashes; add `--source-root /Users/harjaschatha/Final_Project` to verify
against the original source bytes. Full inference requires the original system,
its data, model artifacts, and runtime environment.
