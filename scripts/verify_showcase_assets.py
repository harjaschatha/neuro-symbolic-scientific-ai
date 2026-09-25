#!/usr/bin/env python3
"""Verify imported file hashes, compressed replay identity, and per-case linkage."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from render_confusion_matrices import check_original

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optionally compare imports with original source bytes')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'data/showcase_manifest.json').read_text())
    cases = json.loads((ROOT/'results/verified/cases.json').read_text())
    for entry in manifest['assets']:
        path = ROOT/entry['destination']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['artifact_sha256']:
            raise ValueError(f'Changed imported asset: {path}')
        raw = gzip.decompress(data) if entry['encoding'] == 'gzip' else data
        if hashlib.sha256(raw).hexdigest() != entry['source_sha256']:
            raise ValueError(f'Changed decompressed source: {path}')
        if args.source_root and (args.source_root/entry['source']).read_bytes() != raw:
            raise ValueError(f'Source differs: {entry["source"]}')
        if path.suffix == '.gz':
            name = path.name.removesuffix('.json.gz')
            rows = json.loads(raw)
            compact = [dict(run_id=r['run_id'], true_mechanism=r['true_mechanism'],
                            best_mechanism=r['best_mechanism'], correct_selection=r['correct_selection'],
                            hypothesis_source=r['llm_hypothesis'].get('source', 'unspecified'),
                            residual_source=r.get('residual_source', 'unspecified')) for r in rows]
            if compact != cases[name]:
                raise ValueError(f'Replay and verified cases disagree: {name}')
    if len(manifest['synthetic_samples']) != 28:
        raise ValueError('Expected 28 paired scenario examples')
    for entry in manifest['synthetic_samples']:
        with (ROOT/entry['csv']).open() as f:
            rows = list(csv.DictReader(f))
        if not rows or any(r['run_id'] != entry['name'] for r in rows):
            raise ValueError('Scenario ID mismatch')
        times = [float(r['time_h']) for r in rows]
        if times != sorted(times):
            raise ValueError('Scenario time order mismatch')
    check_original(cases)
    print(f'Verified {len(manifest["assets"])} imported assets, 28 sample pairs, replay linkage, and original matrix counts.')


if __name__ == '__main__':
    main()
