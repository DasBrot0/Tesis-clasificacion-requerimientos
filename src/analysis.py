"""Comparisons across the finished runs of a condition (METODOLOGIA §4.3.2 and G9).

Everything is computed from the pooled outer test predictions that the runs already saved, so it
needs no training and no torch:
- Cochran's Q and pairwise exact McNemar (Holm) between the models of a dataset, per label;
- co-confusion between subcategories and the list of misclassified requirements;
- the best model of each dataset (top Scott-Knott ESD group + Security recall);
- Transformers against Deep Learning models across the three datasets;
- the diversity of the synthetic requirements (C1).
Each function writes its CSV to outputs/output_raw/ and returns the table for the workbook.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .config import (AUGMENTATION_DIR, DATASET_NAMES, DATASETS, LABEL_COLS, MODEL_DISPLAY, MODEL_FAMILY,
                     MODELS, OUTPUT_RAW, SUBCATEGORY_COLS, TEXT_COL, run_dir)
from .significance import cliffs_delta, cochran_q, delta_magnitude, holm, mcnemar_exact

ALPHA = 0.05
NO_SUBCATEGORY = "Ninguna"
SYNTHETIC_DIVERSITY_CSV = "synthetic_diversity.csv"


def load_pooled(condition: str, dataset: str, model_name: str) -> pd.DataFrame | None:
    path = run_dir(condition, dataset, model_name) / "pooled_outer_test_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})
    return frame.sort_values("ID").reset_index(drop=True)


def _matrix(pooled: pd.DataFrame, kind: str, labels: list[str] = LABEL_COLS) -> np.ndarray:
    return pooled[[f"{label}_{kind}" for label in labels]].to_numpy(dtype=int)


def _dataset_predictions(condition: str, dataset: str) -> dict[str, pd.DataFrame]:
    """Pooled predictions of every finished model of a dataset, checked to cover the same IDs."""
    frames = {m: f for m in MODELS if (f := load_pooled(condition, dataset, m)) is not None}
    reference = next(iter(frames.values()), None)
    for model_name, frame in frames.items():
        if not frame["ID"].equals(reference["ID"]):
            raise AssertionError(f"{condition}/{dataset}: {model_name} no evaluó los mismos requisitos")
    return frames


def _active(condition: str, dataset: str, model_name: str) -> np.ndarray:
    summary = json.loads((run_dir(condition, dataset, model_name) / "summary.json").read_text(encoding="utf-8"))
    return np.array([label in summary["labels_in_macro_average"] for label in LABEL_COLS])


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """F1 of each column; 0 when a label has neither positives nor predictions (zero_division=0)."""
    tp = (y_true & y_pred).sum(axis=0)
    denominator = y_true.sum(axis=0) + y_pred.sum(axis=0)
    return np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)


def _save(table: pd.DataFrame, path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return table


# --------------------------------------------------------------------------- Cochran Q + McNemar

def model_tests(condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset and label: Cochran's Q over the hits of every model and, below it, exact McNemar
    for each pair of models with Holm's correction over the pairs of that dataset and label.
    A pair is significant only when Q is (p < 0.05) and its Holm-adjusted p is below 0.05."""
    omnibus, pairs = [], []
    for dataset in DATASETS:
        frames = _dataset_predictions(condition, dataset)
        if len(frames) < 2:
            continue
        names = list(frames)
        y = _matrix(frames[names[0]], "true")
        active = _active(condition, dataset, names[0])
        hits = {m: _matrix(f, "pred") == y for m, f in frames.items()}
        f1 = {m: _f1(y, _matrix(f, "pred")) for m, f in frames.items()}
        for j, label in enumerate(LABEL_COLS):
            if len(names) >= 3:
                q, p_q = cochran_q(np.column_stack([hits[m][:, j] for m in names]))
            else:  # with two models McNemar is already the omnibus test
                q, p_q = np.nan, np.nan
            omnibus.append({
                "dataset": DATASET_NAMES[dataset], "label": label, "in_macro_average": bool(active[j]),
                "support": int(y[:, j].sum()), "models": len(names), "cochran_q": q, "p": p_q,
                "significant": bool(p_q < ALPHA) if len(names) >= 3 else None,
                **{f"accuracy {MODEL_DISPLAY[m]}": float(hits[m][:, j].mean()) for m in names},
            })
            block = []
            for a, b in combinations(names, 2):
                only_a, only_b, p_value = mcnemar_exact(hits[a][:, j], hits[b][:, j])
                block.append({
                    "dataset": DATASET_NAMES[dataset], "label": label, "model_a": MODEL_DISPLAY[a],
                    "model_b": MODEL_DISPLAY[b], "f1_a": float(f1[a][j]), "f1_b": float(f1[b][j]),
                    "only_a_right": only_a, "only_b_right": only_b, "p": p_value,
                })
            for row, adjusted in zip(block, holm([r["p"] for r in block])):
                gate = len(names) < 3 or p_q < ALPHA
                row["p_holm"] = float(adjusted)
                row["significant"] = bool(gate and adjusted < ALPHA)
                row["more_hits"] = (row["model_a"] if row["only_a_right"] > row["only_b_right"] else
                                    row["model_b"] if row["only_b_right"] > row["only_a_right"] else "empate")
            pairs.extend(block)
    omnibus, pairs = pd.DataFrame(omnibus), pd.DataFrame(pairs)
    _save(omnibus, OUTPUT_RAW / condition / "model_tests_cochran_q.csv")
    _save(pairs, OUTPUT_RAW / condition / "model_tests_mcnemar.csv")
    return omnibus, pairs


# --------------------------------------------------------------------------- subcategory confusion

def subcategory_confusion(pooled: pd.DataFrame) -> pd.DataFrame:
    """Rows: real subcategory; columns: predicted subcategory.

    Diagonal: the real subcategory was predicted (TP). Off the diagonal (i, j): the requirement has
    i and not j, the model missed i and predicted j instead, i.e. it took i for j. Column "Ninguna":
    i was missed and no wrong subcategory was predicted in its place. Row "Ninguna": requirements
    without any real subcategory (including non-security ones); there each cell counts a wrongly
    predicted subcategory and "Ninguna" the requirements correctly left without one.
    """
    true, pred = _matrix(pooled, "true", SUBCATEGORY_COLS), _matrix(pooled, "pred", SUBCATEGORY_COLS)
    wrong = (pred == 1) & (true == 0)
    columns = [*SUBCATEGORY_COLS, NO_SUBCATEGORY]
    table = pd.DataFrame(0, index=columns, columns=columns)
    for i, real in enumerate(SUBCATEGORY_COLS):
        missed = (true[:, i] == 1) & (pred[:, i] == 0)
        table.loc[real, real] = int(((true[:, i] == 1) & (pred[:, i] == 1)).sum())
        for j, predicted in enumerate(SUBCATEGORY_COLS):
            if j != i:
                table.loc[real, predicted] = int((missed & wrong[:, j]).sum())
        table.loc[real, NO_SUBCATEGORY] = int((missed & ~wrong.any(axis=1)).sum())
    none_real = true.sum(axis=1) == 0
    for j, predicted in enumerate(SUBCATEGORY_COLS):
        table.loc[NO_SUBCATEGORY, predicted] = int((none_real & (pred[:, j] == 1)).sum())
    table.loc[NO_SUBCATEGORY, NO_SUBCATEGORY] = int((none_real & (pred.sum(axis=1) == 0)).sum())
    return table


def confusion_tables(condition: str) -> pd.DataFrame:
    """Long table (dataset, model, real, predicted, count) of every run's subcategory confusion."""
    parts = []
    for dataset in DATASETS:
        for model_name, pooled in _dataset_predictions(condition, dataset).items():
            long = subcategory_confusion(pooled).rename_axis("real").reset_index().melt(
                id_vars="real", var_name="predicted", value_name="count")
            long.insert(0, "model", MODEL_DISPLAY[model_name])
            long.insert(0, "dataset", DATASET_NAMES[dataset])
            parts.append(long)
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _save(table, OUTPUT_RAW / condition / "subcategory_confusion.csv")


def _label_text(row: np.ndarray) -> str:
    return ", ".join(label for label, value in zip(LABEL_COLS, row) if value) or "—"


def misclassified(condition: str) -> pd.DataFrame:
    """One row per real requirement that at least one model got wrong in some label, with the real
    labels, each model's prediction and the text, for the manual error review of §4.3.2."""
    rows = []
    for dataset, path in DATASETS.items():
        frames = _dataset_predictions(condition, dataset)
        if not frames:
            continue
        texts = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str}).set_index("ID")[TEXT_COL]
        any_frame = next(iter(frames.values()))
        y = _matrix(any_frame, "true")
        preds = {m: _matrix(f, "pred") for m, f in frames.items()}
        wrong_models = sum((p != y).any(axis=1).astype(int) for p in preds.values())
        for k in np.flatnonzero(wrong_models):
            rid = any_frame.at[k, "ID"]
            rows.append({
                "dataset": DATASET_NAMES[dataset], "ID": rid, "models_wrong": int(wrong_models[k]),
                "real": _label_text(y[k]),
                **{MODEL_DISPLAY[m]: _label_text(p[k]) for m, p in preds.items()},
                "requirement": texts.get(rid, ""),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["dataset", "models_wrong", "ID"], ascending=[True, False, True])
    return _save(table, OUTPUT_RAW / condition / "misclassified.csv")


# --------------------------------------------------------------------------- best model

def best_models(condition: str, scott_knott: pd.DataFrame) -> pd.DataFrame:
    """Rule of METODOLOGIA §4.3.2: the models of the top Scott-Knott ESD group do not differ in
    macro-F1 (Binkhonain & Alfayez, 2025); among them, the one with the highest pooled Security
    recall is preferred, because missing a security requirement costs more than a false alarm
    (Petrillo et al., 2025). Macro-F1 breaks an exact tie in recall."""
    rows = []
    for dataset in DATASETS:
        frames = _dataset_predictions(condition, dataset)
        ranking = scott_knott[scott_knott["dataset"] == DATASET_NAMES[dataset]] if not scott_knott.empty else None
        if not frames or ranking is None or ranking.empty:
            continue
        display_to_model = {MODEL_DISPLAY[m]: m for m in frames}
        candidates = [display_to_model[name] for name in ranking.loc[ranking["rank"] == 1, "model"]]
        any_model = next(iter(frames))
        y, active = _matrix(frames[any_model], "true"), _active(condition, dataset, any_model)
        preds = {m: _matrix(frames[m], "pred") for m in candidates}
        macro = {m: float(_f1(y, preds[m])[active].mean()) for m in candidates}
        recall = {m: float((y[:, 0] & preds[m][:, 0]).sum() / max(y[:, 0].sum(), 1)) for m in candidates}
        best = max(candidates, key=lambda m: (recall[m], macro[m]))
        leader = max(candidates, key=lambda m: (macro[m], recall[m]))
        if len(candidates) == 1:
            reason = "único modelo del grupo superior de Scott-Knott ESD"
        elif best == leader:
            reason = "mayor recall de Security del grupo superior (y mayor macro-F1)"
        else:
            reason = (f"mayor recall de Security del grupo superior; {MODEL_DISPLAY[leader]} tiene más macro-F1, "
                      "pero la diferencia no es significativa")
        rows.append({
            "dataset": DATASET_NAMES[dataset], "best_model": MODEL_DISPLAY[best],
            "top_group": " = ".join(MODEL_DISPLAY[m] for m in candidates),
            "macro_f1": macro[best], "security_recall": recall[best], "reason": reason,
            "candidates": "; ".join(f"{MODEL_DISPLAY[m]}: macro-F1 {macro[m]:.3f}, recall Security {recall[m]:.3f}"
                                    for m in sorted(candidates, key=lambda m: -recall[m])),
        })
    return _save(pd.DataFrame(rows), OUTPUT_RAW / condition / "best_model.csv")


# --------------------------------------------------------------------------- Transformers vs DL

def family_comparison(condition: str, runs: list[tuple[str, str, dict]],
                      scott_knott: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transformers (BETO, RoBERTa-BNE) against Deep Learning (CNN, BiLSTM).

    By dataset: the 10 outer-fold macro-F1 of each family (2 models x 5 folds), compared with
    Cliff's delta and Mann-Whitney U, plus each family's best Scott-Knott rank. By label: the mean
    pooled F1 of each family. The last row of each table pools the three datasets."""
    transformers, deep = families = ["Transformers", "Deep Learning"]
    by_dataset, by_label = [], []
    for dataset in DATASETS:
        reports = {m: r for d, m, r in runs if d == dataset}
        members = {f: [m for m in reports if MODEL_FAMILY[m] == f] for f in families}
        if not all(members.values()):
            continue
        folds = {f: np.array([fold["test"]["macro_f1"] for m in members[f] for fold in reports[m]["outer_folds"]])
                 for f in families}
        pooled = {f: [reports[m]["pooled_outer_test"]["macro_f1"] for m in members[f]] for f in families}
        delta = cliffs_delta(folds[transformers], folds[deep])
        ranks = scott_knott[scott_knott["dataset"] == DATASET_NAMES[dataset]].set_index("model")["rank"]
        best_rank = {f: int(min(ranks[MODEL_DISPLAY[m]] for m in members[f])) for f in families}
        by_dataset.append({
            "dataset": DATASET_NAMES[dataset],
            **{f"macro_f1 {f} (media agrupada)": float(np.mean(pooled[f])) for f in families},
            **{f"mejor {f}": MODEL_DISPLAY[max(members[f], key=lambda m: reports[m]["pooled_outer_test"]["macro_f1"])]
               for f in families},
            "delta_macro_f1": float(np.mean(pooled[transformers]) - np.mean(pooled[deep])),
            "cliffs_delta_folds": delta, "magnitude": delta_magnitude(delta),
            "mannwhitney_p": float(mannwhitneyu(folds[transformers], folds[deep]).pvalue),
            **{f"mejor rango SK {f}": best_rank[f] for f in families},
        })
        per_label = {m: {r["label"]: r for r in reports[m]["pooled_outer_test_per_label"]} for m in reports}
        for label in LABEL_COLS:
            f1 = {f: [per_label[m][label]["f1"] for m in members[f]] for f in families}
            by_label.append({
                "dataset": DATASET_NAMES[dataset], "label": label,
                "support": int(per_label[members[transformers][0]][label]["support"]),
                **{f"F1 {f}": float(np.mean(f1[f])) for f in families},
                "delta_f1": float(np.mean(f1[transformers]) - np.mean(f1[deep])),
            })
    by_dataset, by_label = pd.DataFrame(by_dataset), pd.DataFrame(by_label)
    if not by_dataset.empty:
        wins = int((by_dataset["delta_macro_f1"] > 0).sum())
        by_dataset = pd.concat([by_dataset, pd.DataFrame([{
            "dataset": f"Los {len(by_dataset)} datasets",
            **{f"macro_f1 {f} (media agrupada)": by_dataset[f"macro_f1 {f} (media agrupada)"].mean() for f in families},
            "delta_macro_f1": by_dataset["delta_macro_f1"].mean(),
            "magnitude": f"Transformers superiores en {wins} de {len(by_dataset)}",
        }])], ignore_index=True)
        mean_by_label = by_label.groupby("label", sort=False)[[f"F1 {f}" for f in families] + ["delta_f1"]].mean()
        by_label = pd.concat([by_label, mean_by_label.reset_index().assign(
            dataset=f"Media de los {by_dataset.shape[0] - 1} datasets")], ignore_index=True)
    _save(by_dataset, OUTPUT_RAW / condition / "family_comparison.csv")
    _save(by_label, OUTPUT_RAW / condition / "family_comparison_by_label.csv")
    return by_dataset, by_label


# --------------------------------------------------------------------------- synthetic diversity

def synthetic_diversity() -> pd.DataFrame:
    """G9: LaBSE similarity of the accepted synthetics, per dataset.

    The similarities to their source and to the nearest real requirement come from the filter
    (Sim_source, Sim_max_real in synthetic_<dataset>.csv). The mean pairwise similarities among
    synthetics and against the real security requirements need the embeddings and come from
    scripts/augmentation/g9_diversity.py, when it has been run."""
    rows = []
    for dataset in DATASETS:
        path = AUGMENTATION_DIR / f"synthetic_{dataset}.csv"
        if not path.exists():
            continue
        synthetic = pd.read_csv(path, encoding="utf-8-sig")
        rows.append({
            "dataset": DATASET_NAMES[dataset], "synthetics": len(synthetic),
            "sources": synthetic["Anchor_ID"].nunique(),
            "variants_per_source": len(synthetic) / synthetic["Anchor_ID"].nunique(),
            "sim_source_mean": synthetic["Sim_source"].mean(), "sim_source_max": synthetic["Sim_source"].max(),
            "sim_max_real_mean": synthetic["Sim_max_real"].mean(),
            "sim_max_real_max": synthetic["Sim_max_real"].max(),
        })
    table = pd.DataFrame(rows)
    embeddings = OUTPUT_RAW / "synthetic_diversity_labse.csv"
    if not table.empty and embeddings.exists():
        table = table.merge(pd.read_csv(embeddings, encoding="utf-8-sig"), on="dataset", how="left")
    return _save(table, OUTPUT_RAW / SYNTHETIC_DIVERSITY_CSV)
