"""Shared, leakage-safe protocol for the four security multilabel notebooks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support

SEED = 42
DATA_PATH = Path("dataset_seguridad_provisional.csv")
SPLIT_PATH = Path("artifacts/security_split_manifest_seed42.csv")
CV_FOLDS_PATH = Path("artifacts/security_cv_folds_seed42.csv")
N_SPLITS = 5
TEXT_COL = "Requirement_ES"
LABEL_COLS = [
    "Security",
    "Confidentiality",
    "Integrity",
    "Non_repudiation",
    "Accountability",
    "Authenticity",
    "Resistance",
]
SUBCATEGORY_COLS = LABEL_COLS[1:]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    if mapped.isna().any():
        bad = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"{name} contiene valores booleanos no reconocidos: {bad[:10]}")
    return mapped.astype(bool)


def _text_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_security_dataset(
    data_path: Path = DATA_PATH,
    allow_provisional_labels: bool = True,
) -> pd.DataFrame:
    """Load and strictly validate the rows that may enter the experiment."""
    if not data_path.exists():
        raise FileNotFoundError(f"No existe {data_path.resolve()}")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    required = {"ID", TEXT_COL, "Ready_for_training", *LABEL_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df["ID"].isna().any() or df["ID"].astype(str).duplicated().any():
        raise ValueError("ID debe ser completo y único.")

    df["Ready_for_training"] = _as_bool(df["Ready_for_training"], "Ready_for_training")
    df = df[df["Ready_for_training"]].copy()
    if df.empty:
        raise ValueError("No hay filas con Ready_for_training == True.")

    df[TEXT_COL] = _text_key(df[TEXT_COL])
    if df[TEXT_COL].eq("").any() or df[TEXT_COL].str.lower().eq("nan").any():
        raise ValueError("Todas las filas utilizadas deben tener texto no vacío.")
    for col in LABEL_COLS:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"{col} contiene NaN o valores no numéricos.")
        if not converted.isin([0, 1]).all():
            bad = sorted(converted[~converted.isin([0, 1])].unique().tolist())
            raise ValueError(f"{col} debe ser binaria 0/1; valores inválidos: {bad}")
        df[col] = converted.astype(np.int8)

    invalid_hierarchy = df["Security"].eq(0) & df[SUBCATEGORY_COLS].ne(0).any(axis=1)
    if invalid_hierarchy.any():
        raise ValueError("Si Security == 0, las seis subcategorías deben ser 0.")

    if "Is_synthetic" in df.columns:
        df["Is_synthetic"] = _as_bool(df["Is_synthetic"], "Is_synthetic")
    else:
        df["Is_synthetic"] = False

    if not allow_provisional_labels:
        if "Subcategory_label_status" not in df.columns:
            raise ValueError("Falta Subcategory_label_status para exigir validación experta.")
        allowed = {"expert_validated", "consensus"}
        invalid = df["Security"].eq(1) & ~df["Subcategory_label_status"].isin(allowed)
        if invalid.any():
            counts = df.loc[invalid, "Subcategory_label_status"].value_counts(dropna=False).to_dict()
            raise ValueError(
                "ALLOW_PROVISIONAL_LABELS=False: hay positivos de Security sin estado "
                f"expert_validated/consensus: {counts}"
            )

    df["_text_key"] = _text_key(df[TEXT_COL])
    conflicting = df.groupby("_text_key", sort=False)[LABEL_COLS].nunique().max(axis=1).gt(1)
    if conflicting.any():
        raise ValueError("Hay textos exactos duplicados con etiquetas contradictorias.")
    return df.reset_index(drop=True)


def _fallback_multilabel_indices(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic label-aware fallback; less faithful than iterative stratification."""
    rng = np.random.default_rng(seed)
    n = len(y)
    n_test = int(round(n * test_size))
    n_test = min(max(n_test, 1), n - 1)
    desired = y.sum(axis=0) * (n_test / n)
    chosen: list[int] = []
    remaining = set(range(n))
    tie = rng.random(n)
    counts = np.zeros(y.shape[1], dtype=float)
    while len(chosen) < n_test:
        deficits = desired - counts
        scores = []
        for idx in remaining:
            gain = np.maximum(deficits, 0) @ y[idx]
            overshoot = np.maximum(counts + y[idx] - desired, 0).sum()
            scores.append((gain - 0.05 * overshoot, -tie[idx], idx))
        idx = max(scores)[2]
        chosen.append(idx)
        remaining.remove(idx)
        counts += y[idx]
    return np.array(sorted(remaining), dtype=int), np.array(sorted(chosen), dtype=int)


def _multilabel_split(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        keep, held = next(splitter.split(np.zeros(len(y)), y))
        return keep, held, "iterative-stratification"
    except ImportError:
        keep, held = _fallback_multilabel_indices(y, test_size=test_size, seed=seed)
        return keep, held, "deterministic-label-aware-fallback"


def create_split_manifest(data: pd.DataFrame, split_path: Path = SPLIT_PATH) -> pd.DataFrame:
    """Create a deterministic 70/15/15 split with usable rare-label holdouts.

    A constrained integer assignment is used for the current reviewed dataset so
    validation and test each contain at least two Non-repudiation, three
    Accountability and five Resistance examples. Synthetic rows are never
    eligible for validation/test. If the exact target is not applicable, the
    function falls back to iterative multilabel stratification.
    """
    real_data = data[~data["Is_synthetic"]].copy()
    current_totals = real_data[LABEL_COLS].sum().to_numpy(dtype=int)
    expected_totals = np.array([256, 72, 52, 9, 12, 88, 22])
    if len(real_data) == 899 and np.array_equal(current_totals, expected_totals) and not real_data["_text_key"].duplicated().any():
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp

            n = len(real_data)
            sizes = np.array([629, 135, 135])
            targets = np.array([
                [178, 50, 36, 5, 6, 62, 12],
                [39, 11, 8, 2, 3, 13, 5],
                [39, 11, 8, 2, 3, 13, 5],
            ])
            y = real_data[LABEL_COLS].to_numpy(dtype=float)
            rows, bounds = [], []
            for i in range(n):
                row = np.zeros(3 * n)
                row[i] = row[n + i] = row[2 * n + i] = 1
                rows.append(row); bounds.append(1)
            for split_index in range(3):
                row = np.zeros(3 * n); row[split_index*n:(split_index+1)*n] = 1
                rows.append(row); bounds.append(int(sizes[split_index]))
                for label_index in range(len(LABEL_COLS)):
                    row = np.zeros(3 * n)
                    row[split_index*n:(split_index+1)*n] = y[:, label_index]
                    rows.append(row); bounds.append(int(targets[split_index, label_index]))
            objective = np.random.default_rng(SEED).random(3 * n) * 1e-5
            result = milp(
                objective,
                integrality=np.ones(3 * n),
                bounds=Bounds(0, 1),
                constraints=LinearConstraint(np.vstack(rows), bounds, bounds),
            )
            if not result.success:
                raise RuntimeError(result.message)
            assignment = np.rint(result.x.reshape(3, n)).astype(bool)
            split_names = np.full(n, "", dtype=object)
            for split_index, split_name in enumerate(("train", "val", "test")):
                split_names[assignment[split_index]] = split_name
            split_by_key = dict(zip(real_data["_text_key"], split_names))
            split_by_key.update({key: "train" for key in data.loc[data["Is_synthetic"], "_text_key"]})
            method = "constrained-milp-rare-holdout-seed42"
            manifest = pd.DataFrame({
                "ID": data["ID"],
                "split": data["_text_key"].map(split_by_key),
                "is_synthetic": data["Is_synthetic"].astype(bool),
                "text_sha256": data["_text_key"].map(_text_hash),
                "seed": SEED,
                "stratification_method": method,
            })
            split_path.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(split_path, index=False, encoding="utf-8-sig")
            return manifest
        except (ImportError, RuntimeError) as exc:
            print(f"AVISO: no se pudo aplicar el split restringido ({exc}); se usa estratificación iterativa.")

    # Generic fallback for a different future dataset.
    group_rows = []
    for key, part in data.groupby("_text_key", sort=True):
        group_rows.append({
            "_text_key": key,
            "Is_synthetic": bool(part["Is_synthetic"].any()),
            **{label: int(part[label].iloc[0]) for label in LABEL_COLS},
        })
    groups = pd.DataFrame(group_rows)
    real = groups[~groups["Is_synthetic"]].reset_index(drop=True)
    synthetic = groups[groups["Is_synthetic"]].copy()
    if len(real) < 3:
        raise ValueError("Se necesitan al menos tres grupos de texto reales para train/val/test.")

    train_idx, temp_idx, method1 = _multilabel_split(real[LABEL_COLS].to_numpy(), 0.30, SEED)
    temp = real.iloc[temp_idx].reset_index(drop=True)
    val_rel, test_rel, method2 = _multilabel_split(temp[LABEL_COLS].to_numpy(), 0.50, SEED + 1)
    split_by_key = {key: "train" for key in real.iloc[train_idx]["_text_key"]}
    split_by_key.update({key: "val" for key in temp.iloc[val_rel]["_text_key"]})
    split_by_key.update({key: "test" for key in temp.iloc[test_rel]["_text_key"]})
    split_by_key.update({key: "train" for key in synthetic["_text_key"]})
    method = method1 if method1 == method2 else f"{method1}+{method2}"

    manifest = pd.DataFrame({
        "ID": data["ID"],
        "split": data["_text_key"].map(split_by_key),
        "is_synthetic": data["Is_synthetic"].astype(bool),
        "text_sha256": data["_text_key"].map(_text_hash),
        "seed": SEED,
        "stratification_method": method,
    })
    split_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(split_path, index=False, encoding="utf-8-sig")
    return manifest


def get_or_create_splits(data: pd.DataFrame, split_path: Path = SPLIT_PATH):
    expected_ids = set(data["ID"].astype(str))
    if split_path.exists():
        manifest = pd.read_csv(split_path, encoding="utf-8-sig")
        needed = {"ID", "split", "is_synthetic", "text_sha256", "seed", "stratification_method"}
        if needed - set(manifest.columns):
            raise ValueError(f"El manifiesto existente es incompatible; faltan {sorted(needed-set(manifest.columns))}.")
        if set(manifest["ID"].astype(str)) != expected_ids or len(manifest) != len(data):
            raise ValueError("El manifiesto no corresponde exactamente al dataset actual; no se regenera silenciosamente.")
        expected_hash = dict(zip(data["ID"].astype(str), data["_text_key"].map(_text_hash)))
        actual_hash = dict(zip(manifest["ID"].astype(str), manifest["text_sha256"].astype(str)))
        if expected_hash != actual_hash:
            raise ValueError("El texto del dataset cambió respecto al manifiesto existente.")
    else:
        manifest = create_split_manifest(data, split_path)

    if not set(manifest["split"]).issubset({"train", "val", "test"}) or set(manifest["split"]) != {"train", "val", "test"}:
        raise ValueError("El manifiesto debe contener exactamente train, val y test.")
    synth = _as_bool(manifest["is_synthetic"], "manifest.is_synthetic")
    if (synth & manifest["split"].isin(["val", "test"])).any():
        raise ValueError("Un registro sintético aparece en validation o test.")
    if manifest.groupby("text_sha256")["split"].nunique().gt(1).any():
        raise ValueError("Hay textos duplicados exactos distribuidos entre splits.")

    merged = data.merge(manifest[["ID", "split"]], on="ID", how="inner", validate="one_to_one")
    parts = tuple(merged[merged["split"].eq(name)].copy() for name in ("train", "val", "test"))
    if any(part.empty for part in parts):
        raise ValueError("Alguna partición quedó vacía.")
    return (*parts, manifest)


def get_cv_pool_and_test(data: pd.DataFrame, split_path: Path = SPLIT_PATH):
    """Return the old train+validation rows as the CV pool and the untouched test.

    Pre-existing synthetic rows are excluded: augmentation must be regenerated
    from each fold's training partition after splitting.
    """
    train, validation, test, manifest = get_or_create_splits(data, split_path)
    cv_pool = pd.concat([train, validation], ignore_index=True)
    cv_pool = cv_pool[~cv_pool["Is_synthetic"]].copy().reset_index(drop=True)
    test = test[~test["Is_synthetic"]].copy().reset_index(drop=True)
    if set(cv_pool["ID"].astype(str)) & set(test["ID"].astype(str)):
        raise AssertionError("El pool de CV y test comparten IDs.")
    return cv_pool, test, manifest


def create_cv_fold_manifest(
    cv_pool: pd.DataFrame,
    output_path: Path = CV_FOLDS_PATH,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create a deterministic iterative multilabel fold assignment.

    Exact duplicate texts are treated as one group, so related originals never
    straddle train/validation inside a fold. The production protocol requires
    iterative-stratification and deliberately has no ordinary KFold fallback.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "CV multietiqueta requiere iterative-stratification>=0.1.9; "
            "instale requirements-security.txt."
        ) from exc
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")
    if cv_pool.empty or cv_pool["Is_synthetic"].any():
        raise ValueError("El pool de CV debe contener exclusivamente originales.")

    grouped_rows = []
    for text_key, part in cv_pool.groupby("_text_key", sort=True):
        grouped_rows.append({
            "_text_key": text_key,
            **{label: int(part[label].iloc[0]) for label in LABEL_COLS},
        })
    groups = pd.DataFrame(grouped_rows)
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_key: dict[str, int] = {}
    placeholder = np.zeros(len(groups), dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(placeholder, groups[LABEL_COLS].to_numpy(dtype=np.int8))
    ):
        for text_key in groups.iloc[validation_indices]["_text_key"]:
            fold_by_key[str(text_key)] = fold
    manifest = pd.DataFrame({
        "ID": cv_pool["ID"].astype(str),
        "text_sha256": cv_pool["_text_key"].map(_text_hash),
        "fold": cv_pool["_text_key"].map(fold_by_key).astype(int),
        "seed": seed,
        "n_splits": n_splits,
        "stratification_method": "MultilabelStratifiedKFold",
    })
    validate_cv_fold_manifest(cv_pool, manifest, n_splits=n_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    return manifest


def validate_cv_fold_manifest(
    cv_pool: pd.DataFrame, manifest: pd.DataFrame, n_splits: int = N_SPLITS
) -> None:
    """Fail closed if folds do not provide exact, leakage-safe OOF coverage."""
    required = {"ID", "text_sha256", "fold", "seed", "n_splits", "stratification_method"}
    if required - set(manifest.columns):
        raise ValueError(f"Manifiesto CV incompatible: faltan {sorted(required-set(manifest.columns))}.")
    expected_ids = cv_pool["ID"].astype(str)
    actual_ids = manifest["ID"].astype(str)
    if len(manifest) != len(cv_pool) or actual_ids.duplicated().any() or set(actual_ids) != set(expected_ids):
        raise AssertionError("Cada ID original debe aparecer exactamente una vez en el manifiesto CV.")
    if set(pd.to_numeric(manifest["fold"]).astype(int)) != set(range(n_splits)):
        raise AssertionError(f"Los folds deben ser exactamente 0..{n_splits-1}.")
    expected_hash = dict(zip(expected_ids, cv_pool["_text_key"].map(_text_hash)))
    actual_hash = dict(zip(actual_ids, manifest["text_sha256"].astype(str)))
    if expected_hash != actual_hash:
        raise AssertionError("Los textos del pool cambiaron respecto al manifiesto CV.")
    if manifest.groupby("text_sha256")["fold"].nunique().gt(1).any():
        raise AssertionError("Un texto original duplicado aparece en más de un fold.")


def get_or_create_cv_fold_manifest(
    cv_pool: pd.DataFrame,
    output_path: Path = CV_FOLDS_PATH,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> pd.DataFrame:
    if output_path.exists():
        manifest = pd.read_csv(output_path, encoding="utf-8-sig", dtype={"ID": str})
        validate_cv_fold_manifest(cv_pool, manifest, n_splits=n_splits)
        stored_seed = set(pd.to_numeric(manifest["seed"]).astype(int))
        stored_splits = set(pd.to_numeric(manifest["n_splits"]).astype(int))
        if stored_seed != {seed} or stored_splits != {n_splits}:
            raise ValueError("El manifiesto CV existente usa otra semilla o número de folds.")
        return manifest
    return create_cv_fold_manifest(cv_pool, output_path, n_splits, seed)


def fold_distribution_frame(cv_pool: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Long-form train/validation label counts for auditability."""
    fold_by_id = dict(zip(manifest["ID"].astype(str), manifest["fold"].astype(int)))
    assigned = cv_pool.copy()
    assigned["_cv_fold"] = assigned["ID"].astype(str).map(fold_by_id)
    rows = []
    for fold in sorted(assigned["_cv_fold"].unique()):
        for partition, part in (
            ("train", assigned[assigned["_cv_fold"].ne(fold)]),
            ("validation", assigned[assigned["_cv_fold"].eq(fold)]),
        ):
            for label in LABEL_COLS:
                rows.append({
                    "fold": int(fold), "partition": partition, "label": label,
                    "rows": len(part), "positives": int(part[label].sum()),
                    "prevalence": float(part[label].mean()),
                })
    return pd.DataFrame(rows)


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | Iterable[float] = 0.50,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(LABEL_COLS):
        raise ValueError(f"Se esperaban matrices (n, {len(LABEL_COLS)}) con formas iguales.")
    thresholds = np.asarray(threshold, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.repeat(float(thresholds), len(LABEL_COLS))
    if thresholds.shape != (len(LABEL_COLS),):
        raise ValueError(f"threshold debe ser escalar o contener {len(LABEL_COLS)} valores.")
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(np.mean(precision)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "security_recall": float(recall[0]),
        "per_label": pd.DataFrame({
            "label": LABEL_COLS, "precision": precision, "recall": recall, "f1": f1, "support": support,
        }),
        "y_pred": y_pred,
        "threshold": thresholds.tolist(),
    }


def select_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: Iterable[float] = tuple(np.arange(0.40, 0.6001, 0.025)),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Choose each label threshold on validation only, maximizing label F1 then recall."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows: list[dict[str, Any]] = []
    selected: list[float] = []
    for index, label in enumerate(LABEL_COLS):
        label_rows = []
        for candidate in candidates:
            pred = (y_prob[:, index] >= float(candidate)).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, index], pred, average="binary", zero_division=0
            )
            label_rows.append({
                "label": label,
                "threshold": float(candidate),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            })
        label_frame = pd.DataFrame(label_rows).sort_values(
            ["f1", "recall", "threshold"], ascending=[False, False, True], kind="stable"
        )
        winner = label_frame.iloc[0]
        selected.append(float(winner["threshold"]))
        rows.extend(label_rows)
    return np.asarray(selected, dtype=float), pd.DataFrame(rows)


def is_better(current: dict[str, Any], best: dict[str, Any] | None, atol: float = 1e-12) -> bool:
    if best is None:
        return True
    if current["macro_f1"] > best["macro_f1"] + atol:
        return True
    return abs(current["macro_f1"] - best["macro_f1"]) <= atol and current["security_recall"] > best["security_recall"] + atol


def prediction_frame(ids: Iterable[Any], metrics: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame({"ID": list(ids)})
    for j, label in enumerate(LABEL_COLS):
        frame[f"{label}_true"] = metrics["y_true"][:, j].astype(int)
        frame[f"{label}_prob"] = metrics["y_prob"][:, j]
        frame[f"{label}_pred"] = metrics["y_pred"][:, j].astype(int)
    return frame


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
