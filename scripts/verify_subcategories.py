#!/usr/bin/env python3
"""Verifica que las seis subcategorías de seguridad estén presentes en cada dataset de entrenamiento.

Escribe datasets/training/subcategory_coverage.csv y termina con código 1 si alguna subcategoría
está AUSENTE (0 positivos) o es INSUFICIENTE (< MIN_POSITIVES_PER_LABEL) en algún dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    DATASET_NAMES, DATASETS, MIN_POSITIVES_PER_LABEL, SUBCATEGORY_COLS, SUBCATEGORY_DEFINITIONS,
    TRAINING_DIR,
)
from src.data import label_coverage, load_dataset  # noqa: E402


def main() -> int:
    parts = []
    for dataset in DATASETS:
        coverage = label_coverage(load_dataset(dataset))
        coverage.insert(0, "dataset", DATASET_NAMES[dataset])
        parts.append(coverage)
    coverage = pd.concat(parts, ignore_index=True)
    coverage["definition"] = coverage["label"].map(SUBCATEGORY_DEFINITIONS).fillna("security requirement (general)")
    coverage.to_csv(TRAINING_DIR / "subcategory_coverage.csv", index=False, encoding="utf-8-sig")

    print(f"Positivos por etiqueta (mínimo para evaluarla: {MIN_POSITIVES_PER_LABEL})\n")
    print(coverage.pivot(index="label", columns="dataset", values="positives")
          .loc[["Security", *SUBCATEGORY_COLS]].to_string())
    problems = coverage[coverage["label"].isin(SUBCATEGORY_COLS) & coverage["status"].ne("OK")]
    if problems.empty:
        print("\nOK: las seis subcategorías tienen cobertura suficiente en los tres datasets.")
        return 0
    print("\nSubcategorías sin cobertura suficiente:")
    for row in problems.itertuples():
        print(f"  [{row.status}] {row.dataset}: {row.label} = {row.positives} positivos")
    return 1


if __name__ == "__main__":
    sys.exit(main())
