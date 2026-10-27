"""Fast protocol tests; these never train on or predict the frozen test set."""

import unittest

import numpy as np

from security_cross_validation import CONFIGS, fold_frames, verify_frozen_test
from security_multilabel_common import (
    CV_FOLDS_PATH, DATA_PATH, LABEL_COLS, N_SPLITS, SPLIT_PATH,
    get_cv_pool_and_test, get_or_create_cv_fold_manifest, load_security_dataset,
    multilabel_metrics,
)
from security_train_augmentation import augment_balanced_subcategories


class CrossValidationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_security_dataset(DATA_PATH)
        cls.pool, cls.test, _ = get_cv_pool_and_test(data, SPLIT_PATH)
        cls.manifest = get_or_create_cv_fold_manifest(cls.pool, CV_FOLDS_PATH)

    def test_test_is_frozen_and_disjoint(self):
        self.assertEqual(len(self.test), 135)
        self.assertTrue(verify_frozen_test(self.test))
        self.assertFalse(set(self.pool["ID"].astype(str)) & set(self.test["ID"].astype(str)))

    def test_each_original_is_validation_exactly_once(self):
        self.assertEqual(len(self.manifest), len(self.pool))
        self.assertFalse(self.manifest["ID"].astype(str).duplicated().any())
        self.assertEqual(set(self.manifest["fold"].astype(int)), set(range(N_SPLITS)))

    def test_no_text_leakage_and_full_oof_slots(self):
        validation_ids = []
        for fold in range(N_SPLITS):
            train, validation = fold_frames(self.pool, self.manifest, fold)
            self.assertFalse(set(train["_text_key"]) & set(validation["_text_key"]))
            validation_ids.extend(validation["ID"].astype(str))
        self.assertEqual(len(validation_ids), len(self.pool))
        self.assertEqual(set(validation_ids), set(self.pool["ID"].astype(str)))

    def test_augmentation_uses_fold_train_sources_only(self):
        train, validation = fold_frames(self.pool, self.manifest, 0)
        augmented = augment_balanced_subcategories(train, output_path=None)
        generated = augmented[augmented["Is_augmented"]]
        self.assertTrue(set(generated["Augmentation_source_ID"].astype(str)) <= set(train["ID"].astype(str)))
        self.assertFalse(set(generated["ID"].astype(str)) & set(validation["ID"].astype(str)))

    def test_metrics_and_four_models_registered(self):
        y = self.pool[LABEL_COLS].to_numpy(dtype=int)
        metrics = multilabel_metrics(y, y.astype(float), 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0)
        self.assertEqual(set(CONFIGS), {"cnn", "bilstm", "beto", "roberta_bne"})


if __name__ == "__main__":
    unittest.main()
