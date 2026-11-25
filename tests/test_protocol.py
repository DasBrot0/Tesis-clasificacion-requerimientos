"""Fast protocol tests; they never train a model.

    python -m unittest discover -s tests -v

The nested-CV protocol is checked for the two current tasks (binary and subcategories). The joint
task (former five-output model) keeps the tests that do not need its folds: its fold manifests in
outputs/output_raw/splits/ date from 27-10 and no longer match the datasets of 21-11.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    DATASETS, DEFAULT_CONFIGS, LABEL_COLS, MODELS, N_INNER, N_OUTER, NEW_TASKS, SEARCH_SPACES,
    SUBCATEGORY_COLS, TASK_DATASETS, task_spec,
)
from src.data import (  # noqa: E402
    active_label_mask, get_inner_folds, get_outer_folds, load_dataset, load_synthetic, split_by_fold,
    synthetic_path, task_uses_synthetic, with_synthetic,
)
from src.metrics import (  # noqa: E402
    apply_coherence, is_better, multilabel_metrics, rescore_pooled, select_per_label_thresholds,
)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = {}
        for task in NEW_TASKS:
            for dataset in DATASETS:
                data = load_dataset(dataset, task)
                synthetic = (load_synthetic(dataset, data, task)
                             if synthetic_path(dataset).exists() and task_uses_synthetic(task) else None)
                cls.protocol[task, dataset] = (data, get_outer_folds(data, dataset, task), synthetic)

    def nested(self, task, dataset):
        """Yield (outer, outer_train, outer_test, inner, inner_train, inner_val)."""
        data, outer_folds, _ = self.protocol[task, dataset]
        for outer in range(N_OUTER):
            outer_train, outer_test = split_by_fold(data, outer_folds, outer)
            inner_folds = get_inner_folds(outer_train, dataset, outer, task)
            for inner in range(N_INNER):
                yield (outer, outer_train, outer_test, inner, *split_by_fold(outer_train, inner_folds, inner))

    def test_task_files_are_derived_from_the_full_table(self):
        for dataset, path in DATASETS.items():
            full = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str}).set_index("ID")
            binary = load_dataset(dataset, "binary").set_index("ID")
            subcategories = load_dataset(dataset, "subcategories").set_index("ID")
            self.assertEqual(list(binary.index), list(full.index), dataset)
            self.assertTrue(binary["Security"].astype(int).equals(full["Security"].astype(int)))
            self.assertEqual(set(subcategories.index), set(full.index[full["Security"].eq(1)]), dataset)
            self.assertTrue(subcategories[SUBCATEGORY_COLS].astype(int).equals(
                full.loc[subcategories.index, SUBCATEGORY_COLS].astype(int)))
            # each file holds only the labels of its task
            self.assertNotIn("Confidentiality", pd.read_csv(TASK_DATASETS["binary"][dataset], nrows=1).columns)
            self.assertNotIn("Security", pd.read_csv(TASK_DATASETS["subcategories"][dataset], nrows=1).columns)

    def test_joint_dataset_is_hierarchical(self):
        for dataset in DATASETS:
            data = load_dataset(dataset, "joint")
            self.assertEqual(set(data["Dataset"]), {dataset})
            self.assertFalse((data["Security"].eq(0) & data[SUBCATEGORY_COLS].any(axis=1)).any())

    def test_every_row_is_tested_exactly_once(self):
        for (task, dataset), (data, outer_folds, _) in self.protocol.items():
            tested = []
            for outer in range(N_OUTER):
                train, test = split_by_fold(data, outer_folds, outer)
                self.assertFalse(set(train["_text_key"]) & set(test["_text_key"]))
                tested.extend(test["ID"])
            self.assertEqual(sorted(tested), sorted(data["ID"]), f"{task}/{dataset}")

    def test_inner_folds_stay_inside_the_outer_training_part(self):
        for task, dataset in self.protocol:
            for outer, outer_train, outer_test, inner, inner_train, inner_val in self.nested(task, dataset):
                self.assertFalse(set(inner_train["ID"]) & set(outer_test["ID"]))
                self.assertFalse(set(inner_val["ID"]) & set(outer_test["ID"]))
                self.assertEqual(len(inner_train) + len(inner_val), len(outer_train))

    def test_synthetic_rows_never_leak_into_evaluation(self):
        for (task, dataset), (data, outer_folds, synthetic) in self.protocol.items():
            if synthetic is None:
                continue
            self.assertTrue(synthetic["Security"].eq(1).all())
            real_security = data if task == "subcategories" else data[data["Security"].eq(1)]
            self.assertTrue(synthetic["Anchor_ID"].isin(set(real_security["ID"])).all())
            for outer, outer_train, outer_test, inner, inner_train, inner_val in self.nested(task, dataset):
                for train, held in ((outer_train, outer_test), (inner_train, inner_val)):
                    augmented = with_synthetic(train, synthetic)
                    added = augmented.iloc[len(train):]
                    self.assertTrue(added["Anchor_ID"].isin(set(train["ID"])).all())
                    self.assertFalse(added["Anchor_ID"].isin(set(held["ID"])).any())
                    self.assertFalse(held["ID"].str.startswith("syn_").any())
                    self.assertFalse(set(added["_text_key"]) & set(held["_text_key"]))

    def test_scored_labels_have_positives_in_every_partition(self):
        for (task, dataset), (data, _, _) in self.protocol.items():
            labels = list(task_spec(task).labels)
            active = np.array(labels)[active_label_mask(data, task)]
            if task == "binary":
                self.assertEqual(list(active), ["Security"])
            for outer, outer_train, outer_test, inner, inner_train, inner_val in self.nested(task, dataset):
                for part in (outer_test, inner_train, inner_val):
                    self.assertTrue((part[active].sum() > 0).all(), f"{task}/{dataset} {outer}/{inner}")

    def test_binary_metrics(self):
        y = np.array([[1], [1], [0], [0], [1]])
        prob = np.array([[0.9], [0.2], [0.7], [0.1], [0.6]])
        metrics = multilabel_metrics(y, prob, 0.5, task="binary")
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_macro_2class"], (2 / 3 + 0.5) / 2)
        self.assertEqual(list(metrics["per_label"]["label"]), ["Security"])
        thresholds, _ = select_per_label_thresholds(y, prob, np.array([True]), task="binary")
        self.assertEqual(thresholds.shape, (1,))

    def test_subcategory_metrics_ignore_inactive_labels(self):
        data = load_dataset("dosspre", "subcategories")
        y = data[SUBCATEGORY_COLS].to_numpy(dtype=int)
        active = active_label_mask(data, "subcategories")
        metrics = multilabel_metrics(y, y.astype(float), 0.5, active, "subcategories")
        for name in task_spec("subcategories").metrics:
            self.assertAlmostEqual(metrics[name], 1.0, msg=name)
        self.assertEqual(len(metrics["per_label"]), len(SUBCATEGORY_COLS))

    def test_joint_metrics_ignore_inactive_labels(self):
        data = load_dataset("dosspre", "joint")
        y = data[LABEL_COLS].to_numpy(dtype=int)
        active = active_label_mask(data)
        metrics = multilabel_metrics(y, y.astype(float), 0.5, active)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(metrics["per_label"]), len(LABEL_COLS))
        thresholds, _ = select_per_label_thresholds(y, y.astype(float), active)
        self.assertTrue((thresholds[~active] == 0.5).all())

    def test_is_better_uses_the_task_objective_and_tiebreak(self):
        best = {"f1": 0.7, "recall": 0.6, "loss": 0.5}
        self.assertTrue(is_better({"f1": 0.7, "recall": 0.8, "loss": 0.9}, best, task="binary"))
        self.assertFalse(is_better({"f1": 0.69, "recall": 1.0, "loss": 0.1}, best, task="binary"))
        best = {"macro_f1": 0.5, "macro_recall": 0.5, "loss": 0.5}
        self.assertTrue(is_better({"macro_f1": 0.5, "macro_recall": 0.5, "loss": 0.4}, best, task="subcategories"))

    def test_rescore_only_applies_coherence_in_the_joint_task(self):
        pooled = pd.DataFrame({"ID": ["a", "b"], "outer": [0, 0],
                               **{f"{l}_true": [1, 0] for l in SUBCATEGORY_COLS},
                               **{f"{l}_pred": [1, 1] for l in SUBCATEGORY_COLS}})
        report = {"outer_folds": [{"outer": 0, "test": {"loss": 0.1}}]}
        out = rescore_pooled(report, pooled, np.ones(4, dtype=bool), "subcategories")
        self.assertNotIn("coherence_corrections", report)
        self.assertNotIn("Confidentiality_pred_raw", out)
        self.assertAlmostEqual(report["pooled_outer_test"]["macro_recall"], 1.0)

    def test_coherence_rule_only_switches_subcategories_off(self):
        y_pred = np.array([[0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 0, 0, 0]])
        corrected, changed = apply_coherence(y_pred)
        np.testing.assert_array_equal(corrected, [[0, 0, 0, 0, 0], [1, 1, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.assertEqual(int(changed.sum()), 2)
        np.testing.assert_array_equal(corrected[:, 0], y_pred[:, 0])
        self.assertTrue((corrected <= y_pred).all())

    def test_default_configs_lie_inside_the_search_spaces(self):
        self.assertEqual(set(DEFAULT_CONFIGS), set(MODELS))
        for model in MODELS:
            for name, (kind, *spec) in SEARCH_SPACES[model].items():
                value = DEFAULT_CONFIGS[model][name]
                if kind == "categorical":
                    self.assertIn(value, spec[0], f"{model}.{name}")
                else:
                    self.assertTrue(spec[0] <= value <= spec[1], f"{model}.{name}")


if __name__ == "__main__":
    unittest.main()
