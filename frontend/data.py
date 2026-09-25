"""Local data access for the showcase; no inference or external services."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_LABELS = {
    'stage_e_standard': 'Stage E · standard library',
    'stage_e_expanded': 'Stage E · expanded library',
    'earlier_report': 'Earlier report · provisional provenance',
}


def load_replay(name):
    if name not in RESULT_LABELS:
        raise ValueError('Unknown result set')
    path = ROOT / f'results/replay/{name}.json.gz'
    raw = gzip.decompress(path.read_bytes())
    manifest = json.loads((ROOT/'results/verified/manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest() != manifest['sources'][name]['sha256']:
        raise ValueError('Replay does not match the frozen source hash')
    return json.loads(raw)


def accuracy(rows):
    if not rows:
        raise ValueError('No results')
    summary = json.loads((ROOT/'results/verified/summary.json').read_text())
    family = summary['family_mapping']
    truth = [r.get('true_mechanism') or 'baseline' for r in rows]
    selected = [r['best_mechanism'] for r in rows]
    return dict(total=len(rows), saved=sum(r['correct_selection'] for r in rows),
                exact=sum(t == s for t, s in zip(truth, selected)),
                family=sum(family[t] == family[s] for t, s in zip(truth, selected)))
