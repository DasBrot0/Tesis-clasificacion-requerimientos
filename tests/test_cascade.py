"""Pruebas de la cascada binary -> subcategories (src/cascade.py); no entrenan modelos.

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


def _pred(values):
    return {f"{label}_pred": v for label, v in _sub(values).items()}


class CascadeTests(unittest.TestCase):
    def setUp(self):
        # s1, s2: seguridad; n1, n2: no seguridad.
        self.full = pd.DataFrame([
            {"ID": "s1", "Security": 1, **_sub([1, 0, 0, 0])},
            {"ID": "s2", "Security": 1, **_sub([0, 1, 0, 0])},
            {"ID": "n1", "Security": 0, **_sub([0, 0, 0, 0])},
            {"ID": "n2", "Security": 0, **_sub([0, 0, 0, 0])},
        ])
        # Modelo binario: s1 bien, s2 omitido, n1 falso positivo, n2 bien.
        self.binary = pd.DataFrame({"ID": ["s1", "s2", "n1", "n2"], "outer": [0, 1, 1, 0],
                                    "Security_true": [1, 1, 0, 0], "Security_pred": [1, 0, 1, 0]})
        # Tarea 2: predicciones de los requisitos de seguridad y de los de no seguridad de cada fold.
        self.security = pd.DataFrame([{"ID": "s1", "outer": 0, **_pred([1, 0, 0, 1])},
                                      {"ID": "s2", "outer": 1, **_pred([0, 1, 0, 0])}])
        self.nonsecurity = pd.DataFrame([{"ID": "n1", "outer": 1, **_pred([0, 0, 1, 0])},
                                         {"ID": "n2", "outer": 0, **_pred([1, 0, 0, 0])}])

    def test_cascade_switches_off_subcategories_when_binary_says_no(self):
        frame = cascade_predictions(self.binary, self.security, self.nonsecurity, self.full).set_index("ID")
        pred = frame[[f"{label}_pred" for label in LABEL_COLS]]
        self.assertEqual(pred.loc["s1"].tolist(), [1, 1, 0, 0, 1])
        self.assertEqual(pred.loc["s2"].tolist(), [0, 0, 0, 0, 0])  # omitido por el modelo binario
        self.assertEqual(pred.loc["n1"].tolist(), [1, 0, 0, 1, 0])  # falso positivo: pasa a la tarea 2
        self.assertEqual(pred.loc["n2"].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(frame.loc["s2", "Integrity_true"], 1)

    def test_a_requirement_without_subcategory_prediction_is_rejected(self):
        with self.assertRaises(AssertionError):
            cascade_predictions(self.binary, self.security, self.nonsecurity.iloc[:1], self.full)


if __name__ == "__main__":
    unittest.main()
