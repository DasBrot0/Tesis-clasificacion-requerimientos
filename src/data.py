"""Leakage-safe nested-CV protocol, applied independently to each dataset.

    dataset -> N_OUTER multilabel-stratified outer folds (each requirement is tested once)
            -> inside every outer training part, N_INNER inner folds for Optuna

All assignments are written to outputs/output_raw/splits/ and reused by the four models and by both
conditions, so every model of a dataset sees exactly the same partitions.

Condition c1_augmented adds LLM-generated requirements (inputs/interim/augmentation/
synthetic_<dataset>.csv) to training parts only: a synthetic row enters a training part only if
the real requirement it was generated from (Anchor_ID) is in that part. Validation and test parts
never contain synthetic rows.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import (
    OUTPUT_RAW, AUGMENTATION_DIR, DATASETS, LABEL_COLS, MIN_POSITIVES_PER_LABEL, N_INNER, N_OUTER,
    SEED, SUBCATEGORY_COLS, TEXT_COL,
)

SPLITS_DIR = OUTPUT_RAW / "splits"


def _text_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_labels(df: pd.DataFrame, what: str) -> pd.DataFrame:
    df[TEXT_COL] = _text_key(df[TEXT_COL])
    if df[TEXT_COL].eq("").any() or df[TEXT_COL].str.lower().eq("nan").any():
        raise ValueError(f"{what}: todas las filas deben tener texto no vacío.")
    for col in LABEL_COLS:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any() or not converted.isin([0, 1]).all():
            raise ValueError(f"{what}: {col} debe ser binaria 0/1 y sin NaN.")
        df[col] = converted.astype(np.int8)
    if (df["Security"].eq(0) & df[SUBCATEGORY_COLS].ne(0).any(axis=1)).any():
        raise ValueError(f"{what}: si Security == 0, las subcategorías deben ser 0.")
    df["_text_key"] = df[TEXT_COL]
    if df["_text_key"].duplicated().any():
        raise ValueError(f"{what}: hay textos duplicados.")
    return df


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
    return _validate_labels(df, dataset).reset_index(drop=True)


def synthetic_path(dataset: str):
    return AUGMENTATION_DIR / f"synthetic_{dataset}.csv"


def load_synthetic(dataset: str, data: pd.DataFrame) -> pd.DataFrame:
    """Accepted synthetic requirements of one dataset, checked against its real rows."""
    path = synthetic_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}; ejecute el flujo de PLAN_AUMENTO_DATOS.md")
    syn = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str, "Anchor_ID": str})
    missing = {"ID", "Anchor_ID", TEXT_COL, *LABEL_COLS} - set(syn.columns)
    if missing:
        raise ValueError(f"Sintéticos sin columnas requeridas: {sorted(missing)}")
    syn = _validate_labels(syn, f"sintéticos de {dataset}")
    if syn["ID"].duplicated().any() or syn["ID"].isin(set(data["ID"])).any():
        raise ValueError("Los ID sintéticos deben ser únicos y distintos de los reales.")
    if not syn["Anchor_ID"].isin(set(data["ID"])).all():
        raise ValueError("Todo sintético debe apuntar a un requisito real del dataset.")
    if syn["_text_key"].isin(set(data["_text_key"])).any():
        raise ValueError("Un sintético repite el texto de un requisito real.")
    return syn.reset_index(drop=True)


def with_synthetic(train_frame: pd.DataFrame, synthetic: pd.DataFrame | None) -> pd.DataFrame:
    """Training part plus the synthetic rows whose anchor is in it (none for c0_real)."""
    if synthetic is None:
        return train_frame
    extra = synthetic[synthetic["Anchor_ID"].isin(set(train_frame["ID"]))]
    # Real rows get an empty Anchor_ID, so every training row stays traceable.
    return pd.concat([train_frame, extra], ignore_index=True)


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
    labels = data.set_index("ID")[LABEL_COLS].astype(int)
    stored = manifest.set_index("ID")[LABEL_COLS].astype(int).reindex(labels.index)
    if not labels.equals(stored):
        raise ValueError(f"Las etiquetas cambiaron respecto al manifiesto de {what}; regenere los splits.")


def _folds(frame: pd.DataFrame, path, n_splits: int, what: str) -> pd.DataFrame:
    """Deterministic iterative multilabel fold assignment, stored on first use."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    if path.exists():
        manifest = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})
        _validate_manifest(frame, manifest, what)
    else:
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        y = frame[LABEL_COLS].to_numpy(dtype=np.int8)
        fold = np.full(len(frame), -1, dtype=int)
        for index, (_, held_out) in enumerate(splitter.split(np.zeros(len(frame)), y)):
            fold[held_out] = index
        manifest = pd.DataFrame({
            "ID": frame["ID"], "text_sha256": frame["_text_key"].map(_text_hash), "fold": fold,
            **{label: frame[label].astype(int) for label in LABEL_COLS},
            "seed": SEED, "n_splits": n_splits, "stratification_method": "MultilabelStratifiedKFold",
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False, encoding="utf-8-sig")
    if set(manifest["fold"].astype(int)) != set(range(n_splits)):
        raise AssertionError(f"{what}: los folds deben ser exactamente 0..{n_splits - 1}.")
    return manifest


def get_outer_folds(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return _folds(data, SPLITS_DIR / f"{dataset}_outer_folds_seed{SEED}.csv", N_OUTER, "folds externos")


def get_inner_folds(outer_train: pd.DataFrame, dataset: str, outer: int) -> pd.DataFrame:
    return _folds(outer_train, SPLITS_DIR / f"{dataset}_outer{outer}_inner_folds_seed{SEED}.csv",
                  N_INNER, f"folds internos del externo {outer}")


def split_by_fold(frame: pd.DataFrame, manifest: pd.DataFrame, fold: int):
    """(training part, held-out part) of `frame` for one fold of `manifest`."""
    held_ids = set(manifest.loc[manifest["fold"].astype(int).eq(fold), "ID"].astype(str))
    is_held = frame["ID"].isin(held_ids)
    train_frame, held_frame = frame[~is_held].copy(), frame[is_held].copy()
    if train_frame["_text_key"].isin(set(held_frame["_text_key"])).any():
        raise AssertionError("Leakage de texto entre entrenamiento y evaluación.")
    return train_frame.reset_index(drop=True), held_frame.reset_index(drop=True)


def partition_distribution(data: pd.DataFrame, dataset: str, synthetic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-form label counts of every outer and inner partition, for auditability."""
    rows = []

    def add(partition, outer, inner, part):
        for label in LABEL_COLS:
            rows.append({"partition": partition, "outer": outer, "inner": inner, "label": label,
                         "rows": len(part), "positives": int(part[label].sum()),
                         "synthetic_rows": int(part["ID"].str.startswith("syn_").sum())})

    outer_folds = get_outer_folds(data, dataset)
    for outer in range(N_OUTER):
        outer_train, outer_test = split_by_fold(data, outer_folds, outer)
        add("outer_train", outer, -1, with_synthetic(outer_train, synthetic))
        add("outer_test", outer, -1, outer_test)
        inner_folds = get_inner_folds(outer_train, dataset, outer)
        for inner in range(N_INNER):
            inner_train, inner_val = split_by_fold(outer_train, inner_folds, inner)
            add("inner_train", outer, inner, with_synthetic(inner_train, synthetic))
            add("inner_validation", outer, inner, inner_val)
    return pd.DataFrame(rows)
