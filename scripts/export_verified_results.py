#!/usr/bin/env python3
"""Export minimal evidence from pinned local research artifacts; no inference."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'neuro_symbolic_bioprocess/ns_pipeline_outputs/'
ARCHIVE = PREFIX + 'stage_e_archives/job36212117_seed42/'
SOURCES = {
    'earlier_report': (PREFIX + 'report_500/results.json', '6a9556d3b655d5197c1003849115f003fcba686f200030c559888fa62494d0c2'),
    'earlier_local_ablation': (PREFIX + 'hpc_gated_override_v1/full_neuro_symbolic/results.json', 'd75d1d9a88bc9891ec3b11418c493832d7eda6dbe23aecd487876dd358b795ce'),
    'stage_e_standard': (ARCHIVE + 'results/standard_results.json', 'f4382922ba2a6b58a73e1c6407710f7bc9898af44054d819e2ba8fd935552166'),
    'stage_e_expanded': (ARCHIVE + 'results/expanded_ode_results.json', '8edcec866c2ce4da7d82a81cc4210770255008a84d9400944003a4d47980ec22'),
}
FAMILY = {name: name for name in ('baseline', 'maintenance', 'inhibition', 'switching', 'time_gating', 'yield_drift', 'lag_adaptation', 'death_decay')}
FAMILY.update(product_inhibition='inhibition', oxygen_limitation='inhibition', maintenance_inhibition='inhibition', switching_inhibition='switching')


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def validate(rows):
    if len(rows) != 500 or len({r['run_id'] for r in rows}) != 500:
        raise ValueError('Expected 500 unique runs')
    for r in rows:
        if 'error' in r or type(r['correct_selection']) is not bool:
            raise ValueError('Error row or invalid saved correctness')
        if not r['best_mechanism'] or (r['true_mechanism'] or 'baseline') not in FAMILY:
            raise ValueError('Missing selection or unknown truth label')
        if r['best_mechanism'] not in FAMILY:
            raise ValueError('Unknown selection label')


def summarize(rows):
    validate(rows)
    saved = sum(r['correct_selection'] for r in rows)
    exact = sum((r['true_mechanism'] or 'baseline') == r['best_mechanism'] for r in rows)
    family = sum(FAMILY[r['true_mechanism'] or 'baseline'] == FAMILY[r['best_mechanism']] for r in rows)
    return dict(runs=len(rows), saved_correct=saved, saved_accuracy_percent=round(saved / len(rows) * 100, 4),
                exact_correct=exact, exact_accuracy_percent=round(exact / len(rows) * 100, 4),
                family_correct=family, family_accuracy_percent=round(family / len(rows) * 100, 4),
                hypothesis_sources=dict(Counter(r['hypothesis_source'] for r in rows)),
                residual_sources=dict(Counter(r['residual_source'] for r in rows)))


def build(cases):
    summaries = {name: summarize(rows) for name, rows in sorted(cases.items())}
    standard = {r['run_id']: r for r in cases['stage_e_standard']}
    expanded = {r['run_id']: r for r in cases['stage_e_expanded']}
    if standard.keys() != expanded.keys() or any(standard[k]['true_mechanism'] != expanded[k]['true_mechanism'] for k in standard):
        raise ValueError('Stage E cohorts or truth labels differ')
    paired = {}
    for metric in ('saved', 'exact', 'family'):
        def correct(row):
            if metric == 'saved':
                return row['correct_selection']
            truth, selected = row['true_mechanism'] or 'baseline', row['best_mechanism']
            return truth == selected if metric == 'exact' else FAMILY[truth] == FAMILY[selected]
        paired[metric] = dict(rescued=sum(not correct(standard[k]) and correct(expanded[k]) for k in standard),
                              harmed=sum(correct(standard[k]) and not correct(expanded[k]) for k in standard))
    summary = dict(result_sets=summaries, paired_stage_e=paired,
                   truth_normalization='null or empty true_mechanism means baseline', family_mapping=FAMILY,
                   limitations=['Saved projection-aware targets differ between Stage E arms.',
                                'Earlier report HPC provenance remains unresolved; local ablation is a different artifact.',
                                'Stage E run-time configuration and commit snapshots are not verified.',
                                'Synthetic seed-42 development evidence does not establish real-world generalization.',
                                'No hallucination rate, parameter MAPE, RSS, or latency claim is derived here.'])
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['result_set', 'runs', 'saved_correct', 'saved_accuracy_percent', 'exact_correct', 'exact_accuracy_percent', 'family_correct', 'family_accuracy_percent'])
    for name, s in summaries.items():
        writer.writerow([name] + [round(s[k], 4) if isinstance(s[k], float) else s[k] for k in ('runs', 'saved_correct', 'saved_accuracy_percent', 'exact_correct', 'exact_accuracy_percent', 'family_correct', 'family_accuracy_percent')])
    return {'cases.json': encoded(cases), 'summary.json': encoded(summary), 'summary.csv': stream.getvalue().encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Final_Project directory; required for export or source verification')
    parser.add_argument('--check', action='store_true', help='Verify bundled calculations without writing files')
    args = parser.parse_args()
    destination = ROOT / 'results/verified'
    if args.source_root:
        cases = {}
        for name, (relative, expected) in SOURCES.items():
            raw = (args.source_root / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f'Source hash changed: {relative}; review before updating reporting')
            rows = json.loads(raw)
            validate(rows)
            cases[name] = [dict(run_id=r['run_id'], true_mechanism=r['true_mechanism'],
                                best_mechanism=r['best_mechanism'], correct_selection=r['correct_selection'],
                                hypothesis_source=r['llm_hypothesis'].get('source', 'unspecified'),
                                residual_source=r.get('residual_source', 'unspecified')) for r in rows]
    elif args.check:
        cases = json.loads((destination / 'cases.json').read_text())
    else:
        parser.error('--source-root is required for export')
    if set(cases) != set(SOURCES):
        raise ValueError('Unexpected result sets')
    outputs = build(cases)
    manifest = dict(schema_version=1, source_root_label='Final_Project',
                    sources={k: dict(path=v[0], sha256=v[1]) for k, v in SOURCES.items()},
                    artifacts={k: hashlib.sha256(v).hexdigest() for k, v in outputs.items()},
                    provenance='Hashes verify local artifact identity, not original HPC execution provenance.')
    outputs['manifest.json'] = encoded(manifest)
    for name, payload in outputs.items():
        path = destination / name
        if args.check:
            if path.read_bytes() != payload:
                raise ValueError(f'Bundled artifact differs: {name}')
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print('Verified source hashes and bundled evidence.' if args.check and args.source_root else
          'Verified bundled counts, paired cohorts, and artifact hashes.' if args.check else 'Exported verified evidence.')


if __name__ == '__main__':
    main()
