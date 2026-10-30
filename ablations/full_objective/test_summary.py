import unittest

import pandas as pd
from summarize import METRICS, comparison_tables


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.scores = {
            "full": [0.8, 0.6, 0.9, 90.0],
            "full_no_jepa": [0.4, 1.2, 1.4, 70.0],
            "full_no_sigreg": [0.6, 1.0, 1.2, 80.0],
            "full_no_jepa_no_sigreg": [0.3, 1.4, 1.6, 65.0],
        }
        self.metrics = pd.DataFrame([
            {"variant": name, "stage": stage, "parameter": "ki", "split": split,
             **dict(zip(METRICS, values))}
            for stage in ["backbone", "memory"]
            for split in ["test", "val", "ood40"]
            for name, values in self.scores.items()
        ])

    def test_five_contrasts_and_interaction_direction(self):
        deltas, interactions = comparison_tables(self.metrics)
        self.assertEqual(len(deltas), 30)
        self.assertEqual(len(interactions), 6)
        expected_pairs = {
            "effect_of_jepa_in_full": ("full", "full_no_jepa"),
            "effect_of_sigreg_in_full": ("full", "full_no_sigreg"),
            "effect_of_jepa_without_sigreg": ("full_no_sigreg", "full_no_jepa_no_sigreg"),
            "effect_of_sigreg_without_jepa": ("full_no_jepa", "full_no_jepa_no_sigreg"),
            "effect_of_jepa_and_sigreg": ("full", "full_no_jepa_no_sigreg"),
        }
        for row in deltas.to_dict("records"):
            a, b = expected_pairs[row["contrast"]]
            self.assertEqual((row["a"], row["b"]), (a, b))
            for index, metric in enumerate(METRICS):
                self.assertAlmostEqual(row[metric + "_a_minus_b"],
                                       self.scores[a][index] - self.scores[b][index])
            self.assertEqual(row["role"], "training_overlap" if row["split"] == "val" else "benchmark")
        expected_interactions = dict(R2=0.1, MAE=-0.2, rmse=-0.1, p1mag=5.)
        for row in interactions.to_dict("records"):
            self.assertEqual(row["formula"], "full - full_no_jepa - full_no_sigreg + full_no_jepa_no_sigreg")
            self.assertEqual(row["role"], "training_overlap" if row["split"] == "val" else "benchmark")
            for metric, expected in expected_interactions.items():
                self.assertAlmostEqual(row[metric + "_interaction"], expected)
            subset = deltas[(deltas.stage == row["stage"]) & (deltas.split == row["split"])].set_index("contrast")
            for metric in METRICS:
                self.assertAlmostEqual(row[metric + "_interaction"],
                                       subset.loc["effect_of_jepa_in_full", metric + "_a_minus_b"]
                                       - subset.loc["effect_of_jepa_without_sigreg", metric + "_a_minus_b"])

    def test_incomplete_or_duplicate_cells_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "all four configurations"):
            comparison_tables(self.metrics.iloc[1:])
        duplicated = pd.concat([self.metrics, self.metrics.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "all four configurations"):
            comparison_tables(duplicated)


if __name__ == "__main__":
    unittest.main()
