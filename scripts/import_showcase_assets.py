#!/usr/bin/env python3
"""Import an explicit set of synthetic showcase assets; never read Excel files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from export_verified_results import SOURCES

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root
    entries = []

    def copy(relative, target, compressed=False, expected=None):
        raw = (source / relative).read_bytes()
        if expected and sha(raw) != expected:
            raise ValueError(f'Source hash mismatch: {relative}')
        data = gzip.compress(raw, mtime=0) if compressed else raw
        path = ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(dict(source=relative, destination=target, source_sha256=sha(raw),
                            artifact_sha256=sha(data), encoding='gzip' if compressed else 'identity'))

    samples = []
    for csv_path in sorted((source / 'synthetic_data').glob('scenario_*_*.csv')):
        png_path = csv_path.with_suffix('.png')
        if not png_path.exists():
            raise ValueError(f'Missing corresponding figure: {png_path.name}')
        copy(str(csv_path.relative_to(source)), 'data/synthetic/' + csv_path.name)
        copy(str(png_path.relative_to(source)), 'figures/synthetic/' + png_path.name)
        samples.append(dict(name=csv_path.stem, csv='data/synthetic/' + csv_path.name,
                            png='figures/synthetic/' + png_path.name))
    for name in ('representative_upload_timeseries.csv', 'representative_upload_timeseries.manifest.csv', 'sample_upload_timeseries.csv'):
        copy(name, 'data/synthetic/' + name)
    figures = 'report_evidence/hpc_gated_override_v1/analysis/'
    for name in ('figure_5_2_projected_confusion_matrix.png', 'figure_5_2_projected_confusion_matrix_normalized.png',
                 'projected_fittable_confusion_matrix_counts.csv', 'projected_fittable_confusion_matrix_row_percent.csv'):
        copy(figures + name, 'figures/confusion/earlier_report/' + name)
    for name in ('earlier_report', 'stage_e_standard', 'stage_e_expanded'):
        relative, expected = SOURCES[name]
        copy(relative, f'results/replay/{name}.json.gz', compressed=True, expected=expected)
    frontend_source = 'scripts/31_streamlit_inference_app.py'
    manifest = dict(source_root_label='Final_Project', assets=entries, synthetic_samples=samples,
                    frontend_origin=dict(path=frontend_source, sha256=sha((source/frontend_source).read_bytes())),
                    notes=['Only explicitly selected synthetic CSVs, figures, and saved synthetic result JSONs were read.',
                           'Earlier report matrix belongs to the provisional 89.8% artifact, not Stage E.',
                           'Scenario examples and representative upload samples are separate presentation assets; no case-level linkage to frozen results is inferred.'])
    (ROOT/'data/showcase_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Imported {len(samples)} paired CSV/PNG examples and {len(entries)} total assets.')


if __name__ == '__main__':
    main()
