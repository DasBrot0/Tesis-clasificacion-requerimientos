"""Leakage-safe data protocol, applied independently to each dataset.

    dataset -> 15 % stratified hold-out test + 85 % CV pool -> 5 multilabel-stratified folds

Both assignments are written to artifacts/splits/ and reused by the four models, so every model
of a dataset sees exactly the same partitions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS, DATASETS, LABEL_COLS, MIN_POSITIVES_PER_LABEL, N_SPLITS, SEED,
    SUBCATEGORY_COLS, TEST_SIZE, TEXT_COL,
)

SPLITS_DIR = ARTIFACTS / "splits"


def _text_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load and strictly validate one training dataset."""
    data_path = DATASETS[dataset]
    if not data_path.exists():
        raise FileNotFoundError(f"No existe {data_path}; ejecute scripts/prepare_datasets.py")
    df = pd.read_csv(data_path, encoding="utf-8-sig", dtype={"ID": str})
    missing = {"ID", TEXT_COL, *LABEL_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("ID debe ser completo y único.")

    df[TEXT_COL] = _text_key(df[TEXT_COL])
    if df[TEXT_COL].eq("").any() or df[TEXT_COL].str.lower().eq("nan").any():
        raise ValueError("Todas las filas utilizadas deben tener texto no vacío.")
    for col in LABEL_COLS:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any() or not converted.isin([0, 1]).all():
            raise ValueError(f"{col} debe ser binaria 0/1 y sin NaN.")
        df[col] = converted.astype(np.int8)
    if (df["Security"].eq(0) & df[SUBCATEGORY_COLS].ne(0).any(axis=1)).any():
        raise ValueError("Si Security == 0, las seis subcategorías deben ser 0.")
    df["_text_key"] = df[TEXT_COL]
    if df["_text_key"].duplicated().any():
        raise ValueError("Hay textos duplicados; el dataset de entrenamiento debe estar deduplicado.")
    return df.reset_index(drop=True)


def label_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Positives per label and whether each label has enough of them to be scored."""
    rows = []
    for label in LABEL_COLS:
        positives = int(df[label].sum())
        if positives == 0:
            status = "AUSENTE"
        elif positives < MIN_POSITIVES_PER_LABEL:
            status = "INSUFICIENTE"
        else:
            status = "OK"
        rows.append({"label": label, "positives": positives, "rows": len(df),
                     "prevalence": positives / len(df), "status": status})
    return pd.DataFrame(rows)


def active_label_mask(df: pd.DataFrame) -> np.ndarray:
    """Labels with at least MIN_POSITIVES_PER_LABEL positives; only these enter the averages."""
    return df[LABEL_COLS].sum().to_numpy() >= MIN_POSITIVES_PER_LABEL


def _validate_manifest(data: pd.DataFrame, manifest: pd.DataFrame, what: str) -> None:
    ids = manifest["ID"].astype(str)
    if len(manifest) != len(data) or ids.duplicated().any() or set(ids) != set(data["ID"]):
        raise ValueError(f"El manifiesto de {what} no corresponde exactamente al dataset actual.")
    expected_hash = dict(zip(data["ID"], data["_text_key"].map(_text_hash)))
    if expected_hash != dict(zip(ids, manifest["text_sha256"].astype(str))):
        raise ValueError(f"El texto del dataset cambió respecto al manifiesto de {what}.")


def get_cv_pool_and_test(data: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the CV pool and the held-out test, creating the split manifest on first use."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    split_path = SPLITS_DIR / f"{dataset}_split_seed{SEED}.csv"
    if split_path.exists():
        manifest = pd.read_csv(split_path, encoding="utf-8-sig", dtype={"ID": str})
        _validate_manifest(data, manifest, "split")
    else:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        y = data[LABEL_COLS].to_numpy(dtype=np.int8)
        _, test_indices = next(splitter.split(np.zeros(len(data)), y))
        split = np.full(len(data), "cv", dtype=object)
        split[test_indices] = "test"
        manifest = pd.DataFrame({
            "ID": data["ID"], "split": split, "text_sha256": data["_text_key"].map(_text_hash),
            "seed": SEED, "stratification_method": "MultilabelStratifiedShuffleSplit",
        })
        split_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(split_path, index=False, encoding="utf-8-sig")
    merged = data.merge(manifest[["ID", "split"]], on="ID", how="inner", validate="one_to_one")
    cv_pool = merged[merged["split"].eq("cv")].copy().reset_index(drop=True)
    test = merged[merged["split"].eq("test")].copy().reset_index(drop=True)
    if cv_pool.empty or test.empty or set(cv_pool["ID"]) & set(test["ID"]):
        raise AssertionError("El pool de CV y test deben ser no vacíos y disjuntos.")
    return cv_pool, test


def get_cv_folds(cv_pool: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Deterministic iterative multilabel fold assignment (no ordinary KFold fallback)."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    folds_path = SPLITS_DIR / f"{dataset}_cv_folds_seed{SEED}.csv"
    if folds_path.exists():
        manifest = pd.read_csv(folds_path, encoding="utf-8-sig", dtype={"ID": str})
        _validate_manifest(cv_pool, manifest, "folds")
    else:
        splitter = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        y = cv_pool[LABEL_COLS].to_numpy(dtype=np.int8)
        fold = np.full(len(cv_pool), -1, dtype=int)
        for index, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(cv_pool)), y)):
            fold[validation_indices] = index
        manifest = pd.DataFrame({
            "ID": cv_pool["ID"], "text_sha256": cv_pool["_text_key"].map(_text_hash), "fold": fold,
            "seed": SEED, "n_splits": N_SPLITS, "stratification_method": "MultilabelStratifiedKFold",
        })
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(folds_path, index=False, encoding="utf-8-sig")
    if set(manifest["fold"].astype(int)) != set(range(N_SPLITS)):
        raise AssertionError(f"Los folds deben ser exactamente 0..{N_SPLITS - 1}.")
    return manifest


def fold_frames(cv_pool: pd.DataFrame, manifest: pd.DataFrame, fold: int):
    val_ids = set(manifest.loc[manifest["fold"].astype(int).eq(fold), "ID"].astype(str))
    is_val = cv_pool["ID"].isin(val_ids)
    train_frame, validation_frame = cv_pool[~is_val].copy(), cv_pool[is_val].copy()
    if train_frame["_text_key"].isin(set(validation_frame["_text_key"])).any():
        raise AssertionError("Leakage de texto entre train y validation.")
    return train_frame, validation_frame


def partition_distribution(cv_pool: pd.DataFrame, test: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Long-form label counts of test and of every fold's train/validation, for auditability."""
    parts = [("test", -1, test)]
    for fold in range(N_SPLITS):
        train_frame, validation_frame = fold_frames(cv_pool, manifest, fold)
        parts += [("train", fold, train_frame), ("validation", fold, validation_frame)]
    rows = []
    for partition, fold, part in parts:
        for label in LABEL_COLS:
            rows.append({"partition": partition, "fold": fold, "label": label, "rows": len(part),
                         "positives": int(part[label].sum()), "prevalence": float(part[label].mean())})
    return pd.DataFrame(rows)
