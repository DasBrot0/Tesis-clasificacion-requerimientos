"""Pruebas rápidas del protocolo (datos, folds y métricas); no entrenan modelos.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    DATASETS, DEFAULT_CONFIGS, MODELS, N_INNER, N_OUTER, SEARCH_SPACES, SUBCATEGORY_COLS, TASKS,
)
from src.data import (  # noqa: E402
    add_synthetic, inner_folds, load_full_table, load_synthetic, load_task_data, outer_folds, split,
)
from src.metrics import compute_metrics, is_better, select_thresholds  # noqa: E402


class DataTests(unittest.TestCase):
    def test_task_files_are_derived_from_the_full_table(self):
        for dataset in DATASETS:
            full = load_full_table(dataset).set_index("ID")
            binary = load_task_data(dataset, "binary").set_index("ID")
            subcategories = load_task_data(dataset, "subcategories").set_index("ID")
            self.assertEqual(list(binary.index), list(full.index))
            self.assertTrue(binary["Security"].equals(full["Security"]))
            self.assertEqual(set(subcategories.index), set(full.index[full["Security"].eq(1)]))
            self.assertTrue(subcategories[SUBCATEGORY_COLS].equals(full.loc[subcategories.index, SUBCATEGORY_COLS]))

    def test_synthetics_only_enter_through_their_anchor(self):
        data = load_task_data("promise", "subcategories")
        synthetic = load_synthetic("promise", data)
        train = data.iloc[: len(data) // 2]
        added = add_synthetic(train, synthetic).iloc[len(train):]
        expected = synthetic[synthetic["Anchor_ID"].isin(train["ID"])]
        self.assertEqual(sorted(added["ID"]), sorted(expected["ID"]))


class FoldTests(unittest.TestCase):
    def test_outer_folds_cover_every_requirement_once_and_are_shared(self):
        for dataset in DATASETS:
            folds = outer_folds(dataset)
            self.assertEqual(sorted(folds["fold"].unique()), list(range(N_OUTER)))
            self.assertFalse(folds["ID"].duplicated().any())
            for task in TASKS:
                data = load_task_data(dataset, task)
                tested = pd.concat([split(data, folds, k)[1] for k in range(N_OUTER)])
                self.assertEqual(sorted(tested["ID"]), sorted(data["ID"]))

    def test_nested_partitions_do_not_overlap(self):
        for task in TASKS:
            data = load_task_data("secreq", task)
            folds = outer_folds("secreq")
            for outer in range(N_OUTER):
                outer_train, outer_test = split(data, folds, outer)
                self.assertFalse(set(outer_train["ID"]) & set(outer_test["ID"]))
                folds_inner = inner_folds(outer_train, "secreq", task, outer)
                for k in range(N_INNER):
                    inner_train, inner_val = split(outer_train, folds_inner, k)
                    self.assertFalse(set(inner_train["ID"]) & set(inner_val["ID"]))
                    self.assertFalse(set(inner_val["ID"]) & set(outer_test["ID"]))

    def test_every_outer_test_fold_has_security_requirements(self):
        for dataset in DATASETS:
            data = load_task_data(dataset, "binary")
            folds = outer_folds(dataset)
            for k in range(N_OUTER):
                self.assertGreater(split(data, folds, k)[1]["Security"].sum(), 5)


class MetricTests(unittest.TestCase):
    def test_binary_metrics(self):
        y_true = np.array([[1], [1], [0], [0]])
        y_prob = np.array([[0.9], [0.2], [0.8], [0.1]])
        scores = compute_metrics("binary", y_true, y_prob)
        self.assertAlmostEqual(scores["f1"], 0.5)
        self.assertAlmostEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1_macro_2class"], 0.5)

    def test_subcategory_macro_f1_ignores_inactive_labels(self):
        y_true = np.array([[1, 0, 0, 1], [0, 1, 0, 0]])
        y_prob = np.array([[0.9, 0.1, 0.9, 0.9], [0.1, 0.9, 0.1, 0.1]])
        scores = compute_metrics("subcategories", y_true, y_prob, 0.5, active=[True, True, False, True])
        self.assertAlmostEqual(scores["macro_f1"], 1.0)

    def test_is_better_uses_objective_then_tiebreak_then_loss(self):
        best = {"f1": 0.5, "recall": 0.5, "loss": 1.0}
        self.assertTrue(is_better("binary", {"f1": 0.6, "recall": 0.1, "loss": 9.0}, best))
        self.assertTrue(is_better("binary", {"f1": 0.5, "recall": 0.6, "loss": 9.0}, best))
        self.assertTrue(is_better("binary", {"f1": 0.5, "recall": 0.5, "loss": 0.9}, best))
        self.assertFalse(is_better("binary", {"f1": 0.5, "recall": 0.5, "loss": 1.0}, best))

    def test_thresholds_are_chosen_per_label(self):
        y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 0]] * 5)
        y_prob = np.array([[0.45, 0.30, 0.58, 0.2], [0.30, 0.59, 0.41, 0.2]] * 5)
        thresholds, _ = select_thresholds("subcategories", y_true, y_prob, active=[True, True, True, False])
        self.assertLessEqual(thresholds[0], 0.45)
        self.assertLessEqual(thresholds[1], 0.59)
        self.assertEqual(thresholds[3], 0.5)  # etiqueta no evaluada: umbral por defecto


class ConfigTests(unittest.TestCase):
    def test_default_config_is_inside_the_search_space(self):
        for model in MODELS:
            for name, (kind, *spec) in SEARCH_SPACES[model].items():
                value = DEFAULT_CONFIGS[model][name]
                if kind == "categorical":
                    self.assertIn(value, spec[0], f"{model}.{name}")
                else:
                    self.assertTrue(spec[0] <= value <= spec[1], f"{model}.{name}")


if __name__ == "__main__":
    unittest.main()
