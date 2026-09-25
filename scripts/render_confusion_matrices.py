#!/usr/bin/env python3
"""Recalculate confusion counts from verified cases and render labeled PNGs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from export_verified_results import FAMILY

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'figures/confusion'


def matrix(rows, metric):
    def label(value):
        value = value or 'baseline'
        if metric == 'family':
            return FAMILY[value]
        if metric == 'earlier_projection':
            return 'maintenance' if value == 'switching' else value
        return value
    pairs = [(label(r['true_mechanism']), label(r['best_mechanism'])) for r in rows]
    labels = sorted({v for pair in pairs for v in pair})
    counts = [[sum(t == truth and s == selected for t, s in pairs) for selected in labels] for truth in labels]
    return labels, counts


def csv_bytes(labels, counts):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['True mechanism / selected mechanism', *labels])
    writer.writerows([name, *values] for name, values in zip(labels, counts))
    return stream.getvalue().encode()


def check_original(cases):
    labels, counts = matrix(cases['earlier_report'], 'earlier_projection')
    with (OUTPUT/'earlier_report/projected_fittable_confusion_matrix_counts.csv').open() as f:
        saved = list(csv.reader(f))
    if saved[0][1:] != labels or [r[0] for r in saved[1:]] != labels or [[int(v) for v in r[1:]] for r in saved[1:]] != counts:
        raise ValueError('Original report confusion counts do not match frozen report cases')
    if sum(counts[i][i] for i in range(len(labels))) != 449:
        raise ValueError('Original report diagonal does not match saved report accuracy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    raw = (ROOT/'results/verified/cases.json').read_bytes()
    cases = json.loads(raw)
    check_original(cases)
    if not args.check:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    generated = {}
    for name in ('earlier_report', 'stage_e_standard', 'stage_e_expanded'):
        for metric in (('exact',) if name == 'earlier_report' else ('exact', 'family')):
            labels, counts = matrix(cases[name], metric)
            directory = OUTPUT/name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/f'{metric}_counts.csv'
            payload = csv_bytes(labels, counts)
            if args.check:
                if path.read_bytes() != payload:
                    raise ValueError(f'Counts differ: {path}')
            else:
                path.write_bytes(payload)
                a = np.array(counts)
                size = max(7, len(labels) * .82)
                fig, ax = plt.subplots(figsize=(size+2, size))
                plot = ax.imshow(a, cmap='Blues', vmin=0)
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
                ax.set_yticks(range(len(labels)), labels)
                ax.set_xlabel('Selected mechanism'); ax.set_ylabel('True mechanism')
                accuracy = int(np.trace(a))
                title = name.replace('_', ' ').title()
                caveat = '\nEarlier report: original HPC provenance unresolved' if name == 'earlier_report' else '\nPaired synthetic development cohort; seed 42'
                ax.set_title(f'{title} — {metric} labels\n{accuracy}/500 ({accuracy/5:.1f}%)'+caveat, pad=18)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, str(a[i,j]), ha='center', va='center', fontsize=9,
                                color='white' if a[i,j] > a.max()*.55 else '#18283a')
                fig.colorbar(plot, ax=ax, label='Cases', shrink=.7)
                fig.tight_layout()
                fig.savefig(directory/f'{metric}_counts.png', dpi=160)
                plt.close(fig)
            for p in (path, directory/f'{metric}_counts.png'):
                generated[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(cases_sha256=hashlib.sha256(raw).hexdigest(), artifacts=generated,
                    orientation='Rows: ground truth; columns: selected mechanism. Counts, not percentages.',
                    original_report_check='Copied projected counts equal recomputation from earlier_report: diagonal 449, total 500.')
    path = OUTPUT/'generated_manifest.json'
    if args.check:
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Generated figure hashes differ')
    else:
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    print('Confusion counts and figure integrity verified.' if args.check else 'Rendered five confusion matrices; original projected report matrix verified.')


if __name__ == '__main__':
    main()
