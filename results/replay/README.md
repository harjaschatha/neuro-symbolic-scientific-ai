# Full frozen result payloads for the frontend

These gzip files contain byte-preserving compressed copies of the three source
JSON result files used in the showcase: the earlier provisional report and the
standard/expanded Stage E arms. Each has 500 synthetic cases with fitted-candidate
rankings, hypothesis records, diagnostics, gates, and other fields recorded by
the original pipeline. The UI presents absent fields as absent rather than
reconstructing evidence.

Decompressed SHA-256 hashes match `results/verified/manifest.json`. The compressed
hashes and source paths are in `data/showcase_manifest.json`. The conflicting
75.2% earlier local ablation remains in compact verified evidence, but is not
included as a full frontend replay dataset.

No model weights, raw experimental Excel files, or inference server are needed
for replay. These records preserve the known provenance limitations described in
the main README.
