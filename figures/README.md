# Research figures

## Original synthetic examples

`synthetic/` contains 28 PNGs copied unchanged from the main project's
`synthetic_data/`, paired with CSVs under `data/synthetic/`. These show synthetic
trajectories, baseline fits, and residual patterns. Source hashes are recorded in
`data/showcase_manifest.json`.

## Confusion matrices

`confusion/earlier_report/figure_5_2_projected_confusion_matrix.png` and its
normalized variant are the original report figures. The associated count CSV
was checked cell by cell against the frozen earlier report cases with switching
projected to maintenance: **449/500 (89.8%)** on the diagonal. Original HPC
provenance for that report artifact remains unresolved. This is not a Stage E
matrix and does not represent 89.8% exact biological-label accuracy.

The `exact_counts.png` files show common-label biological correctness for the
earlier report and both Stage E arms. The Stage E `family_counts.png` files apply
the shared family mapping. These five PNGs are newly rendered from frozen saved
labels, with their underlying count CSVs alongside them. No model was rerun.
Rows are truth and columns are selected mechanisms; unavailable labels count as
errors in exact comparison.

Regenerate the new figures with Matplotlib installed:

```bash
python scripts/render_confusion_matrices.py
```

Verify CSV counts, the original projected count matrix, and generated-file hashes
without Matplotlib:

```bash
python scripts/render_confusion_matrices.py --check
```
