# Synthetic showcase data

`synthetic/` contains 28 original scenario CSVs: calibration and validation
examples for 14 scenario configurations, with matching original PNGs in
`figures/synthetic/`. These are the main project's synthetic data illustrations,
not the portfolio's separate four-curve demo.

The sample configurations cover baseline, maintenance, inhibition, high noise,
switching, time gating, maintenance-inhibition, switching-inhibition, lag
adaptation, yield drift, product inhibition, oxygen limitation, death/decay, and
switching-maintenance. A configuration is not necessarily a distinct biological
label; high noise is an observation condition.

Each scenario CSV includes `run_id`, `scenario`, `time_h`, ground-truth channels
(`*_true`), observations (`*_obs`), baseline predictions (`*_baseline`), residuals,
and split labels. Biomass and substrate concentrations in the original figures
are in g/L and time is in hours. The recorded train/test split inside a CSV is
separate from the calibration/validation scenario suffix.

`representative_upload_timeseries.csv`, its selection manifest, and
`sample_upload_timeseries.csv` are the original frontend's synthetic upload
examples. Their historical selection metadata is retained unchanged. These
presentation samples are not asserted to be the exact trajectories underlying
the archived Stage E results; the frontend does not attach their measurements
to a replayed decision by run ID alone.

`showcase_manifest.json` records the original relative paths and hashes of every
imported asset, including figures and compressed replay payloads. No Excel file
or experimental spreadsheet data was imported.

Verify assets and their linkage to the compact verified cases:

```bash
python scripts/verify_showcase_assets.py
```

Add `--source-root /Users/harjaschatha/Final_Project` to compare original bytes.
