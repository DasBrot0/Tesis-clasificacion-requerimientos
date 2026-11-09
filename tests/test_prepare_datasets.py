"""Deduplication tests for scripts/prepare_datasets.py; they never train a model.

    python -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATASETS, LABEL_COLS, TEXT_COL  # noqa: E402

_spec = importlib.util.spec_from_file_location("prepare_datasets", ROOT / "scripts" / "prepare_datasets.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _row(en, es, labels, original_label="X"):
    return {"Requirement_original": en, TEXT_COL: es, "Original_label": original_label,
            "Subcategory_label_status": "structural_zero", "Prelabel_evidence": "",
            **dict(zip(LABEL_COLS, labels))}


class NearDuplicateTests(unittest.TestCase):
    def test_key_ignores_case_accents_punctuation_and_english_articles(self):
        self.assertEqual(prep._dedup_key("Él  deberá, cifrar."), prep._dedup_key("el debera cifrar"))
        self.assertEqual(prep._dedup_key("The SAG system shall perform well in the web environments,", True),
                         prep._dedup_key("The SAG system shall perform well in web environments.", True))
        # sin drop_articles (español) los artículos cuentan; los números no se pegan
        self.assertNotEqual(prep._dedup_key("el sistema"), prep._dedup_key("sistema"))
        self.assertNotEqual(prep._dedup_key("version 1.5"), prep._dedup_key("version 15"))

    def test_near_duplicates_are_dropped_and_ids_kept(self):
        frame = pd.DataFrame([
            _row("Return the appropriate error.", "Devuelve el error.", [0, 0, 0, 0, 0]),
            _row("Other requirement", "Otro requisito", [1, 1, 0, 0, 0]),
            _row("return the appropriate error,", "Devuelve el error correspondiente,", [0, 0, 0, 0, 0]),
            _row("A distinct one", "Devuelve el error", [0, 0, 0, 0, 0]),  # mismo español, sin puntuación
        ])
        out = prep._deduplicate_all(frame, "toy", merge_english_conflicts=False)
        self.assertEqual(list(out["ID"]), ["toy_0001", "toy_0002"])

    def test_contradictory_near_duplicates_follow_the_existing_policy(self):
        frame = pd.DataFrame([
            _row("The system shall sign documents.", "Firmar.", [0, 0, 0, 0, 0], "Functionality"),
            _row("the system shall sign documents", "Firmar documentos", [1, 0, 0, 1, 0], "Authentication"),
        ])
        # PROMISE/SecReq: un conflicto en el inglés es un error, igual que con duplicados exactos
        with self.assertRaises(ValueError):
            prep._deduplicate_all(frame.copy(), "toy", merge_english_conflicts=False)
        # DOSSPRE: unión de etiquetas y estado class_mapped_merged_duplicates
        out = prep._deduplicate_all(frame.copy(), "toy", merge_english_conflicts=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[LABEL_COLS].iloc[0].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(out["Original_label"].iloc[0], "Functionality | Authentication")
        self.assertEqual(out["Subcategory_label_status"].iloc[0], "class_mapped_merged_duplicates")

    def test_training_datasets_have_no_near_duplicates(self):
        for dataset, path in DATASETS.items():
            data = pd.read_csv(path, encoding="utf-8-sig")
            for col, drop_articles in (("Requirement_original", True), (TEXT_COL, False)):
                keys = data[col].dropna().map(lambda v: prep._dedup_key(v, drop_articles))
                self.assertFalse(keys.duplicated().any(), f"{dataset}: casi duplicados en {col}")


if __name__ == "__main__":
    unittest.main()
