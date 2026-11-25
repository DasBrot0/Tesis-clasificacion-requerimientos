"""Carga de los datos y particiones de la validación cruzada anidada.

    tabla completa del dataset
      -> 5 folds externos (estratificados sobre Security y las cuatro subcategorías)
           tarea binary:        todos los requisitos del fold
           tarea subcategories: solo los requisitos de seguridad del fold
      -> dentro del entrenamiento de cada fold externo, 3 folds internos para Optuna

Los folds se guardan en outputs/output_raw/splits/ la primera vez y se reutilizan en todas las
corridas, así que los cinco modelos y las dos condiciones ven exactamente las mismas particiones.

Sintéticos (condición c1_augmented, solo tarea subcategories): un sintético entra en una
partición de entrenamiento solo si el requisito real del que se generó (Anchor_ID) está en ella.
Las particiones de validación y prueba nunca tienen sintéticos.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from .config import (
    LABEL_COLS, MIN_POSITIVES_PER_LABEL, N_INNER, N_OUTER, SEED, SPLITS_DIR, TEXT_COL,
    full_table_path, get_task, synthetic_path, task_file_path,
)

# --------------------------------------------------------------------------- carga


def _check_labels(df: pd.DataFrame, labels: list[str], what: str) -> pd.DataFrame:
    """Texto no vacío y sin duplicados; etiquetas 0/1 sin valores faltantes."""
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if df[TEXT_COL].eq("").any():
        raise ValueError(f"{what}: hay filas sin texto.")
    if df[TEXT_COL].duplicated().any():
        raise ValueError(f"{what}: hay textos duplicados.")
    for col in labels:
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.isin([0, 1]).all():
            raise ValueError(f"{what}: la columna {col} debe ser 0/1.")
        df[col] = values.astype("int64")
    return df


def load_full_table(dataset: str) -> pd.DataFrame:
    """Tabla completa del dataset: todos los requisitos, con Security y las cuatro subcategorías."""
    df = pd.read_csv(full_table_path(dataset), encoding="utf-8-sig", dtype={"ID": str})
    return _check_labels(df, LABEL_COLS, dataset)


def load_task_data(dataset: str, task: str) -> pd.DataFrame:
    """Archivo de entrenamiento de una tarea: <dataset>_binary.csv o <dataset>_subcategories.csv.

    Los requisitos de seguridad sin ninguna subcategoría quedan en la tarea 2 con las cuatro
    etiquetas en 0 (decisión del 24-11-2026).
    """
    path = task_file_path(dataset, task)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}; ejecute scripts/prepare_datasets.py")
    labels = list(get_task(task).labels)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})
    df = _check_labels(df, labels, f"{dataset} ({task})")
    if df["ID"].duplicated().any():
        raise ValueError(f"{dataset} ({task}): hay ID repetidos.")
    return df[["ID", TEXT_COL, *labels]].reset_index(drop=True)


def load_synthetic(dataset: str, real: pd.DataFrame) -> pd.DataFrame:
    """Sintéticos aceptados de un dataset (todos son requisitos de seguridad)."""
    labels = list(get_task("subcategories").labels)
    df = pd.read_csv(synthetic_path(dataset), encoding="utf-8-sig", dtype={"ID": str, "Anchor_ID": str})
    df = _check_labels(df, labels, f"sintéticos de {dataset}")
    if not df["Anchor_ID"].isin(real["ID"]).all():
        raise ValueError("Todo sintético debe apuntar a un requisito real de la tarea.")
    if df["ID"].isin(real["ID"]).any() or df[TEXT_COL].isin(real[TEXT_COL]).any():
        raise ValueError("Un sintético repite el ID o el texto de un requisito real.")
    return df[["ID", "Anchor_ID", TEXT_COL, *labels]].reset_index(drop=True)


def add_synthetic(train: pd.DataFrame, synthetic: pd.DataFrame | None) -> pd.DataFrame:
    """Partición de entrenamiento + los sintéticos cuyo requisito de origen está en ella."""
    if synthetic is None:
        return train
    extra = synthetic[synthetic["Anchor_ID"].isin(train["ID"])]
    return pd.concat([train, extra.drop(columns="Anchor_ID")], ignore_index=True)


# --------------------------------------------------------------------------- etiquetas evaluadas


def active_labels(df: pd.DataFrame, task: str) -> np.ndarray:
    """Etiquetas con al menos MIN_POSITIVES_PER_LABEL positivos: solo estas entran en los promedios."""
    return df[list(get_task(task).labels)].sum().to_numpy() >= MIN_POSITIVES_PER_LABEL


def label_coverage(df: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    for label in get_task(task).labels:
        positives = int(df[label].sum())
        rows.append({"label": label, "positives": positives, "rows": len(df),
                     "prevalence": positives / len(df),
                     "scored": positives >= MIN_POSITIVES_PER_LABEL})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- folds


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assign_folds(frame: pd.DataFrame, labels: list[str], n_splits: int) -> np.ndarray:
    """Número de fold de cada fila. Estratificación simple con una etiqueta, iterativa con varias."""
    y = frame[labels].to_numpy()
    if len(labels) == 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = splitter.split(np.zeros(len(frame)), y[:, 0])
    else:
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = splitter.split(np.zeros(len(frame)), y)
    folds = np.full(len(frame), -1)
    for k, (_, held_out) in enumerate(splits):
        folds[held_out] = k
    return folds


def _load_or_create_folds(frame: pd.DataFrame, labels: list[str], n_splits: int, path) -> pd.DataFrame:
    """Tabla ID -> fold, guardada en `path`. Si ya existe, se comprueba que siga correspondiendo
    a los mismos requisitos (mismo texto y mismas etiquetas)."""
    current = pd.DataFrame({"ID": frame["ID"], "text_sha256": frame[TEXT_COL].map(_text_hash),
                            **{label: frame[label] for label in labels}})
    if path.exists():
        stored = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})
        same = stored.drop(columns="fold").sort_values("ID").reset_index(drop=True)
        if not same.equals(current.sort_values("ID").reset_index(drop=True)):
            raise ValueError(f"Los datos cambiaron respecto a {path.name}; borre outputs/output_raw/splits/.")
        return stored[["ID", "fold"]]
    current["fold"] = _assign_folds(frame, labels, n_splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(path, index=False, encoding="utf-8-sig")
    return current[["ID", "fold"]]


def outer_folds(dataset: str) -> pd.DataFrame:
    """Folds externos del dataset, los mismos para las dos tareas."""
    full = load_full_table(dataset)
    return _load_or_create_folds(full, LABEL_COLS, N_OUTER, SPLITS_DIR / f"{dataset}_outer_folds.csv")


def inner_folds(outer_train: pd.DataFrame, dataset: str, task: str, outer: int) -> pd.DataFrame:
    """Folds internos del entrenamiento de un fold externo, estratificados sobre las etiquetas de la tarea."""
    path = SPLITS_DIR / task / f"{dataset}_outer{outer}_inner_folds.csv"
    return _load_or_create_folds(outer_train, list(get_task(task).labels), N_INNER, path)


def split(frame: pd.DataFrame, folds: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entrenamiento, evaluación) de `frame` para el fold k."""
    held_ids = set(folds.loc[folds["fold"] == k, "ID"])
    is_held = frame["ID"].isin(held_ids)
    return frame[~is_held].reset_index(drop=True), frame[is_held].reset_index(drop=True)
