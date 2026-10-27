"""Fast protocol tests; they never train a model.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    DATASETS, DEFAULT_CONFIGS, LABEL_COLS, MODELS, N_INNER, N_OUTER, SEARCH_SPACES, SUBCATEGORY_COLS,
)
from src.data import (  # noqa: E402
    active_label_mask, get_inner_folds, get_outer_folds, load_dataset, load_synthetic, split_by_fold,
    synthetic_path, with_synthetic,
)
from src.metrics import multilabel_metrics, select_per_label_thresholds  # noqa: E402


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = {}
        for dataset in DATASETS:
            data = load_dataset(dataset)
            synthetic = load_synthetic(dataset, data) if synthetic_path(dataset).exists() else None
            cls.protocol[dataset] = (data, get_outer_folds(data, dataset), synthetic)

    def nested(self, dataset):
        """Yield (outer, outer_train, outer_test, inner, inner_train, inner_val)."""
        data, outer_folds, _ = self.protocol[dataset]
        for outer in range(N_OUTER):
            outer_train, outer_test = split_by_fold(data, outer_folds, outer)
            inner_folds = get_inner_folds(outer_train, dataset, outer)
            for inner in range(N_INNER):
                yield (outer, outer_train, outer_test, inner, *split_by_fold(outer_train, inner_folds, inner))

    def test_datasets_are_separate_and_hierarchical(self):
        for dataset, (data, _, _) in self.protocol.items():
            self.assertEqual(set(data["Dataset"]), {dataset})
            self.assertFalse((data["Security"].eq(0) & data[SUBCATEGORY_COLS].any(axis=1)).any())

    def test_every_row_is_tested_exactly_once(self):
        for dataset, (data, outer_folds, _) in self.protocol.items():
            tested = []
            for outer in range(N_OUTER):
                train, test = split_by_fold(data, outer_folds, outer)
                self.assertFalse(set(train["_text_key"]) & set(test["_text_key"]))
                tested.extend(test["ID"])
            self.assertEqual(sorted(tested), sorted(data["ID"]), dataset)

    def test_inner_folds_stay_inside_the_outer_training_part(self):
        for dataset in DATASETS:
            for outer, outer_train, outer_test, inner, inner_train, inner_val in self.nested(dataset):
                self.assertFalse(set(inner_train["ID"]) & set(outer_test["ID"]))
                self.assertFalse(set(inner_val["ID"]) & set(outer_test["ID"]))
                self.assertEqual(len(inner_train) + len(inner_val), len(outer_train))

    def test_synthetic_rows_never_leak_into_evaluation(self):
        for dataset, (data, outer_folds, synthetic) in self.protocol.items():
            if synthetic is None:
                continue
            self.assertTrue(synthetic["Anchor_ID"].isin(set(data.loc[data["Security"].eq(1), "ID"])).all())
            for outer, outer_train, outer_test, inner, inner_train, inner_val in self.nested(dataset):
                for train, held in ((outer_train, outer_test), (inner_train, inner_val)):
                    augmented = with_synthetic(train, synthetic)
                    added = augmented.iloc[len(train):]
                    self.assertTrue(added["Anchor_ID"].isin(set(train["ID"])).all())
                    self.assertFalse(added["Anchor_ID"].isin(set(held["ID"])).any())
                    self.assertFalse(held["ID"].str.startswith("syn_").any())
                    self.assertFalse(set(added["_text_key"]) & set(held["_text_key"]))

    def test_scored_labels_have_positives_in_every_partition(self):
        for dataset in DATASETS:
            data = self.protocol[dataset][0]
            active = np.array(LABEL_COLS)[active_label_mask(data)]
            self.assertIn("Security", active)
            for outer, outer_train, outer_test, inner, inner_train, inner_val in self.nested(dataset):
                for part in (outer_test, inner_train, inner_val):
                    self.assertTrue((part[active].sum() > 0).all(), f"{dataset} externo {outer} interno {inner}")

    def test_metrics_ignore_inactive_labels(self):
        data = self.protocol["dosspre"][0]
        y = data[LABEL_COLS].to_numpy(dtype=int)
        active = active_label_mask(data)
        metrics = multilabel_metrics(y, y.astype(float), 0.5, active)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(metrics["per_label"]), len(LABEL_COLS))
        thresholds, _ = select_per_label_thresholds(y, y.astype(float), active)
        self.assertTrue((thresholds[~active] == 0.5).all())

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
