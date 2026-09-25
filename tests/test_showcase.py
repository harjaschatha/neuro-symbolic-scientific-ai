"""Verify replay linkage, metric semantics, and navigable showcase pages."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontend.data import load_replay, accuracy


class TestShowcaseEvidence(unittest.TestCase):
    def test_replays_preserve_verified_counts(self):
        expected = {'stage_e_standard': (419, 156), 'stage_e_expanded': (418, 316), 'earlier_report': (449, 375)}
        summaries = json.loads((ROOT/'results/verified/summary.json').read_text())['result_sets']
        for name, (saved, exact) in expected.items():
            with self.subTest(name=name):
                counts = accuracy(load_replay(name))
                self.assertEqual(counts['total'], 500)
                self.assertEqual((counts['saved'], counts['exact']), (saved, exact))
                self.assertEqual(counts['family'], summaries[name]['family_correct'])

    def test_reject_unknown_replay(self):
        with self.assertRaises(ValueError):
            load_replay('../untrusted')


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Optional showcase dependencies not installed')
class TestShowcaseFrontend(unittest.TestCase):
    def start(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        return at

    def test_results_switch_study_and_matrix(self):
        at = self.start()
        self.assertEqual([m.value for m in at.metric], ['500', '83.8%', '31.2%', '49.2%'])
        at.selectbox(key='study').set_value('stage_e_expanded').run()
        self.assertEqual([m.value for m in at.metric], ['500', '83.6%', '63.2%', '67.0%'])
        at.selectbox(key='matrix_metric').set_value('Fixed mechanism families').run()
        self.assertEqual(len(at.exception), 0)
        at.selectbox(key='study').set_value('earlier_report').run()
        self.assertEqual([m.value for m in at.metric][:3], ['500', '89.8%', '75.0%'])
        at.checkbox(key='normalized_matrix').check().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any('provenance' in w.value for w in at.warning))

    def test_case_panels_and_empty_filter(self):
        at = self.start()
        at.radio(key='page').set_value('Case explorer').run()
        for name in ('stage_e_standard', 'stage_e_expanded', 'earlier_report'):
            at.selectbox(key='study').set_value(name).run()
            self.assertEqual(len(at.exception), 0)
            self.assertTrue(any(t.label == 'Audit Log' for t in at.tabs))
        at.selectbox(key='exact_filter').set_value('Incorrect').run()
        self.assertTrue(any('Exact biological label: incorrect' in c.value for c in at.caption))
        at.text_input(key='case_search').set_value('no-such-run-xyz').run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any('No cases match' in info.value for info in at.info))

    def test_sample_gallery_and_csv_preview(self):
        at = self.start()
        at.radio(key='page').set_value('Synthetic data').run()
        at.selectbox(key='sample_name').set_value('scenario_4_switching_cal').run()
        at.checkbox(key='preview_sample').check().run()
        self.assertEqual(len(at.exception), 0)
        self.assertGreater(len(at.selectbox(key='uploaded_preview_run').options), 1)
        self.assertFalse(any('not sorted' in w.value for w in at.warning))


if __name__ == '__main__':
    unittest.main()
