"""Regression checks for report evidence, independent of inference dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_results', ROOT / 'scripts/export_verified_results.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class TestVerifiedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / 'results/verified/cases.json').read_text())

    def test_bundled_counts_and_common_label_metrics(self):
        result = json.loads(exporter.build(self.cases)['summary.json'])
        standard = result['result_sets']['stage_e_standard']
        expanded = result['result_sets']['stage_e_expanded']
        self.assertEqual((standard['saved_correct'], standard['exact_correct'], standard['family_correct']), (419, 156, 246))
        self.assertEqual((expanded['saved_correct'], expanded['exact_correct'], expanded['family_correct']), (418, 316, 335))
        self.assertEqual(result['paired_stage_e']['exact'], {'rescued': 166, 'harmed': 6})

    def test_null_truth_is_baseline(self):
        rows = copy.deepcopy(self.cases['earlier_report'])
        exact = sum((r['true_mechanism'] or 'baseline') == r['best_mechanism'] for r in rows)
        self.assertEqual(exact, 375)
        self.assertEqual(exporter.summarize(rows)['exact_correct'], exact)

    def test_serialization_order_does_not_change_evidence(self):
        reversed_cases = dict(reversed(list(self.cases.items())))
        self.assertEqual(exporter.build(self.cases), exporter.build(reversed_cases))

    def test_reject_duplicate_ids(self):
        rows = copy.deepcopy(self.cases['stage_e_standard'])
        rows[1]['run_id'] = rows[0]['run_id']
        with self.assertRaises(ValueError):
            exporter.summarize(rows)

    def test_reject_mismatched_paired_truth(self):
        cases = copy.deepcopy(self.cases)
        row = cases['stage_e_expanded'][0]
        row['true_mechanism'] = 'maintenance' if row['true_mechanism'] != 'maintenance' else 'baseline'
        with self.assertRaises(ValueError):
            exporter.build(cases)


if __name__ == '__main__':
    unittest.main()
