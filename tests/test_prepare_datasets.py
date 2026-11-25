"""Pruebas de scripts/prepare_datasets.py (duplicados y archivos por tarea); no entrenan modelos.

    python -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATASETS, LABEL_COLS, TEXT_COL, full_table_path  # noqa: E402

_spec = importlib.util.spec_from_file_location("prepare_datasets", ROOT / "scripts" / "prepare_datasets.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _row(en, es, labels, original_label="X"):
    return {"Requirement_original": en, TEXT_COL: es, "Original_label": original_label,
            "Subcategory_label_status": "structural_zero", "Prelabel_evidence": "",
            **dict(zip(LABEL_COLS, labels))}


class DuplicateTests(unittest.TestCase):
    def test_key_ignores_case_accents_punctuation_and_english_articles(self):
        key = prep.near_duplicate_key
        self.assertEqual(key("Él  deberá, cifrar."), key("el debera cifrar"))
        self.assertEqual(key("The SAG system shall perform well in the web environments,", True),
                         key("The SAG system shall perform well in web environments.", True))
        # en español los artículos cuentan; los números no se pegan
        self.assertNotEqual(key("el sistema"), key("sistema"))
        self.assertNotEqual(key("version 1.5"), key("version 15"))

    def test_near_duplicates_are_dropped_and_ids_kept(self):
        frame = pd.DataFrame([
            _row("Return the appropriate error.", "Devuelve el error.", [0, 0, 0, 0, 0]),
            _row("Other requirement", "Otro requisito", [1, 1, 0, 0, 0]),
            _row("return the appropriate error,", "Devuelve el error correspondiente,", [0, 0, 0, 0, 0]),
            _row("A distinct one", "Devuelve el error", [0, 0, 0, 0, 0]),  # mismo español sin puntuación
        ])
        out = prep.remove_duplicates(frame, "toy", merge_conflicts=False)
        self.assertEqual(list(out["ID"]), ["toy_0001", "toy_0002"])

    def test_contradictory_duplicates(self):
        frame = pd.DataFrame([
            _row("The system shall sign documents.", "Firmar.", [0, 0, 0, 0, 0], "Functionality"),
            _row("the system shall sign documents", "Firmar documentos", [1, 0, 0, 1, 0], "Authentication"),
        ])
        # PROMISE y SecReq: un conflicto es un error
        with self.assertRaises(ValueError):
            prep.remove_duplicates(frame.copy(), "toy", merge_conflicts=False)
        # DOSSPRE: unión de etiquetas
        out = prep.remove_duplicates(frame.copy(), "toy", merge_conflicts=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[LABEL_COLS].iloc[0].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(out["Original_label"].iloc[0], "Functionality | Authentication")
        self.assertEqual(out["Subcategory_label_status"].iloc[0], "class_mapped_merged_duplicates")

    def test_training_datasets_have_no_near_duplicates(self):
        for dataset in DATASETS:
            data = pd.read_csv(full_table_path(dataset), encoding="utf-8-sig")
            for col, drop_articles in (("Requirement_original", True), (TEXT_COL, False)):
                keys = data[col].dropna().map(lambda v: prep.near_duplicate_key(v, drop_articles))
                self.assertFalse(keys.duplicated().any(), f"{dataset}: casi duplicados en {col}")


class TaskTableTests(unittest.TestCase):
    def test_task_tables_split_binary_and_subcategories(self):
        cols = {c: "x" for c in prep.COMMON_COLS}
        full = pd.DataFrame([
            {**cols, "ID": "a", "Security": 1, "Confidentiality": 1, "Integrity": 0, "Authenticity": 0,
             "Resistance": 0, "Subcategory_label_status": "llm_provisional", "Prelabel_evidence": "e"},
            {**cols, "ID": "b", "Security": 0, "Confidentiality": 0, "Integrity": 0, "Authenticity": 0,
             "Resistance": 0, "Subcategory_label_status": "structural_zero", "Prelabel_evidence": ""},
        ])
        binary, subcategories = prep.task_tables(full)
        self.assertEqual(list(binary["ID"]), ["a", "b"])
        self.assertEqual(binary.columns[-1], "Security")
        self.assertEqual(list(subcategories["ID"]), ["a"])
        self.assertNotIn("Security", subcategories.columns)


if __name__ == "__main__":
    unittest.main()
