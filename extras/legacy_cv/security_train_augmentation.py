"""Deterministic, train-only augmentation for rare security subcategories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from security_multilabel_common import LABEL_COLS, SEED, SUBCATEGORY_COLS, TEXT_COL

AUGMENTED_TRAIN_PATH = Path("artifacts/security_train_augmented_seed42.csv")

PREFIXES = [
    "Como requisito obligatorio,",
    "Se establece la siguiente condición:",
    "La solución deberá cumplir que",
    "Es necesario garantizar que",
    "El sistema debe asegurar que",
    "Se exige que",
    "Como condición del sistema,",
    "La aplicación deberá garantizar que",
    "El producto debe cumplir lo siguiente:",
    "Se considera obligatorio que",
]
SUFFIXES = [
    "",
    " Esta condición deberá cumplirse durante toda la operación.",
    " El cumplimiento de este requisito será obligatorio.",
    " Esta regla se aplicará en todos los casos pertinentes.",
    " El sistema deberá mantener esta condición de forma continua.",
    " Esta condición formará parte de la operación normal del sistema.",
]


def _variant(text: str, variant_index: int) -> str:
    core = str(text).strip()
    if core and core[-1] not in ".!?":
        core += "."
    prefix = PREFIXES[variant_index % len(PREFIXES)]
    suffix = SUFFIXES[(variant_index // len(PREFIXES)) % len(SUFFIXES)]
    return f"{prefix} {core[0].lower() + core[1:]}{suffix}".strip()


def augment_balanced_subcategories(
    train_df: pd.DataFrame,
    output_path: Path | None = AUGMENTED_TRAIN_PATH,
    seed: int = SEED,
) -> pd.DataFrame:
    """Equalize positive counts of the six subcategories using train-only sources.

    Only single-subcategory sources are used, so increasing one target cannot
    silently overshoot another. Security stays 1 for every augmented row.
    """
    train = train_df.copy().reset_index(drop=True)
    train["Is_augmented"] = False
    train["Augmentation_source_ID"] = pd.NA
    train["Augmentation_variant"] = pd.NA
    target = int(train[SUBCATEGORY_COLS].sum().max())
    existing_texts = set(train[TEXT_COL].astype(str))
    synthetic_rows = []
    rng = np.random.default_rng(seed)

    for label in SUBCATEGORY_COLS:
        deficit = target - int(train[label].sum())
        if deficit <= 0:
            continue
        other_labels = [item for item in SUBCATEGORY_COLS if item != label]
        sources = train[train[label].eq(1) & train[other_labels].sum(axis=1).eq(0)].copy()
        if sources.empty:
            raise ValueError(f"No hay semillas exclusivas de train para balancear {label} sin alterar otras etiquetas.")
        sources = sources.iloc[rng.permutation(len(sources))].reset_index(drop=True)
        made = 0
        variant_index = 0
        while made < deficit:
            source = sources.iloc[made % len(sources)]
            text = _variant(source[TEXT_COL], variant_index)
            variant_index += 1
            if text in existing_texts:
                continue
            row = source.copy()
            row["ID"] = f"aug_{label}_{made+1:04d}"
            row[TEXT_COL] = text
            row["Is_synthetic"] = True
            row["Is_augmented"] = True
            row["Augmentation_source_ID"] = source["ID"]
            row["Augmentation_variant"] = variant_index - 1
            row["split"] = "train"
            synthetic_rows.append(row)
            existing_texts.add(text)
            made += 1

    augmented = pd.concat([train, pd.DataFrame(synthetic_rows)], ignore_index=True)
    counts = augmented[SUBCATEGORY_COLS].sum().astype(int)
    if counts.nunique() != 1:
        raise AssertionError(f"Las subcategorías no quedaron balanceadas: {counts.to_dict()}")
    if not augmented.loc[augmented["Is_augmented"], "Security"].eq(1).all():
        raise AssertionError("Toda fila aumentada debe conservar Security=1.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        augmented.to_csv(output_path, index=False, encoding="utf-8-sig")
    return augmented
