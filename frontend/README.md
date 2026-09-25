# Research workbench showcase

Run from the repository root:

```bash
python -m pip install -r requirements-showcase.txt
python -m streamlit run app.py
```

The frontend runs locally and loads bundled files. It needs neither the private
experimental spreadsheet nor an LLM endpoint, model weights, or the original
`Final_Project` directory. Tested with Streamlit 1.59.0.

- **Results:** choose a frozen study; view the separately defined accuracies,
  confusion matrices, underlying counts, and PNG/CSV downloads.
- **Synthetic data:** browse 28 original scenario CSV/PNG pairs, select plotted
  channels, download samples, and preview a bundled or uploaded CSV.
- **Case explorer:** filter saved decisions and inspect the original workbench's
  diagnostics, hypotheses, critiques, candidate fits, guards, reports, and JSON
  audit views. Correctness filters use exact biological labels; saved
  projection-aware correctness is separately labeled.

`workbench.py` adapts the presentation functions and CSS from
`Final_Project/scripts/31_streamlit_inference_app.py`. That source's hash is
recorded in `data/showcase_manifest.json`. The portfolio retains its evidence
panels, while `app.py` supplies navigation, verified study selection, and a data
and figure gallery. Original live-inference controls were removed because this
repository is a self-contained replay showcase, not the full research backend.
CSV uploads are previews only and do not change saved results.

The copied original report matrix retains its unresolved-provenance note.
Stage E matrices use common-label exact or fixed-family counts. No simulated
benchmark output is displayed as research evidence.
