"""Tests of the binary -> subcategories cascade (src/cascade.py); they never train a model.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cascade import cascade_predictions  # noqa: E402
from src.config import LABEL_COLS, SUBCATEGORY_COLS  # noqa: E402


def _sub(values):
    return dict(zip(SUBCATEGORY_COLS, values))


class CascadeTests(unittest.TestCase):
    def setUp(self):
        # s1, s2: security; n1, n2: not security.
        self.full = pd.DataFrame([
            {"ID": "s1", "Security": 1, **_sub([1, 0, 0, 0])},
            {"ID": "s2", "Security": 1, **_sub([0, 1, 0, 0])},
            {"ID": "n1", "Security": 0, **_sub([0, 0, 0, 0])},
            {"ID": "n2", "Security": 0, **_sub([0, 0, 0, 0])},
        ])
        # Binary model: s1 right, s2 missed, n1 false positive, n2 right.
        self.binary = pd.DataFrame({"ID": ["s1", "s2", "n1", "n2"], "outer": [0, 1, 1, 0],
                                    "Security_true": [1, 1, 0, 0], "Security_pred": [1, 0, 1, 0]})
        self.sub = pd.DataFrame([
            {"ID": "s1", "outer": 0, **{f"{l}_pred": v for l, v in _sub([1, 0, 0, 1]).items()}},
            {"ID": "s2", "outer": 1, **{f"{l}_pred": v for l, v in _sub([0, 1, 0, 0]).items()}},
        ])
        # Every subcategory fold model predicts every non-security requirement.
        self.other = pd.DataFrame([
            {"ID": "n1", "outer": 0, **{f"{l}_pred": v for l, v in _sub([1, 1, 1, 1]).items()}},
            {"ID": "n1", "outer": 1, **{f"{l}_pred": v for l, v in _sub([0, 0, 1, 0]).items()}},
            {"ID": "n2", "outer": 0, **{f"{l}_pred": v for l, v in _sub([1, 0, 0, 0]).items()}},
            {"ID": "n2", "outer": 1, **{f"{l}_pred": v for l, v in _sub([0, 1, 0, 0]).items()}},
        ])

    def test_cascade_switches_off_and_uses_the_matching_fold(self):
        frame = cascade_predictions(self.binary, self.sub, self.other, self.full).set_index("ID")
        pred = frame[[f"{l}_pred" for l in LABEL_COLS]]
        self.assertEqual(pred.loc["s1"].tolist(), [1, 1, 0, 0, 1])
        self.assertEqual(pred.loc["s2"].tolist(), [0, 0, 0, 0, 0])   # missed by the binary model
        self.assertEqual(pred.loc["n1"].tolist(), [1, 0, 0, 1, 0])   # fold 1 model, same as binary fold
        self.assertEqual(pred.loc["n2"].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(frame.loc["s2", "Integrity_true"], 1)

    def test_mismatched_binary_truth_is_rejected(self):
        broken = self.binary.assign(Security_true=[1, 1, 1, 0])
        with self.assertRaises(AssertionError):
            cascade_predictions(broken, self.sub, self.other, self.full)


if __name__ == "__main__":
    unittest.main()
