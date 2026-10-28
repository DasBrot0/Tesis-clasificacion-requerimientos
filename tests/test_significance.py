"""Tests of the statistical tests and the subcategory confusion; they need no training data.

    python -m unittest tests.test_significance -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis import NO_SUBCATEGORY, subcategory_confusion  # noqa: E402
from src.config import LABEL_COLS  # noqa: E402
from src.significance import cochran_q, holm, mcnemar_exact, scott_knott_esd  # noqa: E402


class SignificanceTests(unittest.TestCase):
    def test_cochran_q_with_two_models_is_mcnemar_chi_square(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(200) < 0.7, rng.random(200) < 0.6
        q, _ = cochran_q(np.column_stack([a, b]))
        only_a, only_b, _ = mcnemar_exact(a, b)
        self.assertAlmostEqual(q, (only_a - only_b) ** 2 / (only_a + only_b))

    def test_cochran_q_without_disagreement(self):
        self.assertEqual(cochran_q(np.ones((10, 3))), (0.0, 1.0))

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_scott_knott_groups(self):
        samples = {"a": [0.80, 0.81, 0.82, 0.83, 0.84], "b": [0.80, 0.82, 0.81, 0.84, 0.83],
                   "c": [0.50, 0.51, 0.52, 0.53, 0.54]}
        ranks = scott_knott_esd(samples).set_index("treatment")["rank"]
        self.assertEqual(ranks["a"], ranks["b"])
        self.assertGreater(ranks["c"], ranks["a"])


class SubcategoryConfusionTests(unittest.TestCase):
    def test_swap_omission_and_extra(self):
        rows = [  # (true labels, predicted labels)
            ({"Security", "Integrity"}, {"Security", "Authenticity"}),       # Integrity taken for Authenticity
            ({"Security", "Integrity"}, {"Security", "Integrity"}),          # hit
            ({"Security", "Confidentiality"}, {"Security"}),                 # omitted
            ({"Security"}, {"Security", "Resistance"}),                      # extra subcategory
            (set(), set()),                                                  # correctly none
        ]
        pooled = pd.DataFrame([{**{f"{l}_true": int(l in t) for l in LABEL_COLS},
                                **{f"{l}_pred": int(l in p) for l in LABEL_COLS}} for t, p in rows])
        table = subcategory_confusion(pooled)
        self.assertEqual(table.loc["Integrity", "Authenticity"], 1)
        self.assertEqual(table.loc["Integrity", "Integrity"], 1)
        self.assertEqual(table.loc["Confidentiality", NO_SUBCATEGORY], 1)
        self.assertEqual(table.loc[NO_SUBCATEGORY, "Resistance"], 1)
        self.assertEqual(table.loc[NO_SUBCATEGORY, NO_SUBCATEGORY], 1)
        self.assertEqual(int(table.values.sum()), 5)


if __name__ == "__main__":
    unittest.main()
