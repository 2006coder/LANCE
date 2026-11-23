import unittest

import pandas as pd

from analysis.loss_contrasts import loss_contrasts


class LossContrastTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame({
            "parameter": "kcat",
            "split": "test",
            "stage": "memory",
            "role": "benchmark",
            "variant": ["full", "full_no_jepa", "full_no_sigreg", "full_no_jepa_no_sigreg"],
            "R2": [0.50, 0.53, 0.58, 0.56],
        })

    def test_compares_within_each_regularizer_setting(self):
        scores = pd.concat([
            self.scores,
            self.scores.assign(stage="backbone", R2=0.1),
            self.scores.assign(role="diagnostic", R2=0.9),
        ])
        result = loss_contrasts(scores)
        self.assertEqual(result[["parameter", "split"]].values.tolist(), [["kcat", "test"]])
        self.assertAlmostEqual(result.at[0, "sigreg_retained"], 0.03)
        self.assertAlmostEqual(result.at[0, "sigreg_removed"], -0.02)

    def test_rejects_incomplete_and_duplicate_comparisons(self):
        duplicate = pd.concat([self.scores, self.scores.iloc[[0]]])
        for scores in (self.scores.iloc[:-1], duplicate):
            with self.subTest(rows=len(scores)), self.assertRaises(ValueError):
                loss_contrasts(scores)
