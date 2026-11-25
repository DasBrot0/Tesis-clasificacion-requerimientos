"""Corrida completa de una (tarea, condición, dataset, modelo) con validación cruzada anidada.

Para cada uno de los 5 folds externos:
  1. Optuna ajusta los hiperparámetros con los 3 folds internos del entrenamiento externo.
  2. El umbral de cada etiqueta se elige con las predicciones fuera de fold del mejor trial.
  3. Se reentrena con todo el entrenamiento externo (tantas épocas como la media de las mejores
     épocas internas) y se predice una sola vez el fold de prueba.

Los 5 folds de prueba cubren cada requisito real una vez; sus predicciones juntas dan las métricas
que se reportan. En c1_augmented los sintéticos solo entran en las particiones de entrenamiento.

En la tarea de subcategorías, el modelo de cada fold externo predice también los requisitos sin
seguridad de ese mismo fold de prueba (nunca entrena con ellos). src/cascade.py usa esas
predicciones para evaluar la cascada binary -> subcategories de extremo a extremo.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    DATASET_NAMES, DEFAULT_CONFIGS, MAX_LENGTH, MODEL_DISPLAY, N_INNER, N_OUTER, N_TRIALS, NONSECURITY_CSV,
    SEARCH_SPACES, SEED,
    SPLITS_DIR, TEXT_COL, WORD_MODELS, checkpoint_dir, get_task, run_dir,
)
from .data import (
    active_labels, add_synthetic, inner_folds, label_coverage, load_synthetic, load_task_data, outer_folds, split,
)
from .metrics import add_pooled_scores, compute_metrics, metric_names, prediction_frame, save_json, select_thresholds
from .models import DEVICE
from .reporting import (
    export_excel, save_inner_curves, save_run_confusion, update_comparison_table, write_run_tables,
)
from .training import fit, make_loader, predict, release_memory
from .tuning import trial_files, trials_table, tune




def nonsecurity_requirements(dataset: str) -> pd.DataFrame:
    """Requisitos con Security = 0. El modelo de subcategorías nunca entrena con ellos."""
    binary = load_task_data(dataset, "binary")
    return binary.loc[binary["Security"].eq(0), ["ID", TEXT_COL]].reset_index(drop=True)


def predict_frame(result, model_name: str, frame: pd.DataFrame, labels: list[str], batch_size: int):
    loader = make_loader(frame, labels, result.collator, batch_size)
    use_amp = DEVICE.type == "cuda" and model_name not in WORD_MODELS
    probabilities, loss = predict(result.model, loader, result.loss_fn, use_amp)
    return loader.dataset.ids, probabilities, loss


def save_checkpoint(result, model_name: str, config: dict, thresholds, labels, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"config": config, "labels": labels, "thresholds": dict(zip(labels, thresholds.tolist())), "seed": SEED}
    if model_name in WORD_MODELS:
        torch.save({"state_dict": result.model.state_dict(), "vocab": result.collator.vocab,
                    "max_len": result.collator.max_len, **meta}, directory / "model.pt")
    else:
        result.model.hf.save_pretrained(directory)
        result.collator.tokenizer.save_pretrained(directory)
        meta.update({"source": result.collator.source, "max_length": MAX_LENGTH})
    save_json(directory / "metrics_config.json", meta)


def run_outer_fold(task: str, condition: str, dataset: str, model_name: str, outer: int,
                   data: pd.DataFrame, folds: pd.DataFrame, synthetic: pd.DataFrame | None,
                   active: np.ndarray, n_trials: int, out_dir: Path, save_checkpoints: bool) -> dict:
    spec = get_task(task)
    labels = list(spec.labels)
    fold_dir = out_dir / f"outer_{outer}"
    result_path = fold_dir / "outer_result.json"
    if result_path.exists():  # fold ya terminado en una corrida anterior
        print(f"  externo {outer + 1}/{N_OUTER}: ya completado", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    started = time.time()

    outer_train, outer_test = split(data, folds, outer)
    n_synthetic = len(add_synthetic(outer_train, synthetic)) - len(outer_train)
    print(f"\n--- externo {outer + 1}/{N_OUTER}: entrenamiento={len(outer_train)} reales + {n_synthetic} "
          f"sintéticos | prueba={len(outer_test)} ---", flush=True)

    # 1) Optuna sobre los folds internos
    folds_inner = inner_folds(outer_train, dataset, task, outer)
    study = tune(task, dataset, model_name, outer_train, folds_inner, active, fold_dir, n_trials, synthetic)
    trials = trials_table(study, model_name)
    trials.to_csv(fold_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    winner = trials.iloc[0]
    winner_id = int(winner["trial_id"])
    config = {**DEFAULT_CONFIGS[model_name], **study.trials[winner_id].params}

    # 2) umbrales por etiqueta con las predicciones fuera de fold del mejor trial
    oof_path, history_path = trial_files(fold_dir, winner_id)
    oof = np.load(oof_path)
    if sorted(oof["ids"].tolist()) != sorted(outer_train["ID"].tolist()):
        raise AssertionError("Las predicciones internas no cubren exactamente el entrenamiento externo.")
    thresholds, threshold_search = select_thresholds(task, oof["y_true"], oof["y_prob"], active)
    threshold_search.to_csv(fold_dir / "inner_oof_threshold_search.csv", index=False, encoding="utf-8-sig")
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    history.to_csv(fold_dir / "inner_cv_history.csv", index=False, encoding="utf-8-sig")
    save_inner_curves(task, history, fold_dir / "inner_cv_curves.png",
                      f"{DATASET_NAMES[dataset]} — {MODEL_DISPLAY[model_name]} — externo {outer + 1} ({task}, {condition})")

    # 3) modelo final con todo el entrenamiento externo, y una predicción del fold de prueba
    final_epochs = max(1, int(round(float(oof["best_epochs"].mean()))))
    train = add_synthetic(outer_train, synthetic)
    print(f"  modelo final: {final_epochs} épocas sobre {len(train)} filas, config={config}", flush=True)
    result = fit(task, model_name, config, train, SEED + 100 + outer, epochs=final_epochs, keep_model=True)
    if save_checkpoints:
        save_checkpoint(result, model_name, {**config, "epochs": final_epochs}, thresholds, labels,
                        checkpoint_dir(task, condition, dataset, model_name, outer))

    ids, test_prob, test_loss = predict_frame(result, model_name, outer_test, labels, config["batch_size"])
    test_true = outer_test[labels].to_numpy(dtype=int)
    scores = compute_metrics(task, test_true, test_prob, thresholds, active)
    predictions = prediction_frame(task, ids, test_true, test_prob, scores["y_pred"])
    predictions.insert(1, "outer", outer)
    predictions.to_csv(fold_dir / "outer_test_predictions.csv", index=False, encoding="utf-8-sig")

    if task == "subcategories":  # requisitos sin seguridad del mismo fold de prueba, para la cascada
        others = nonsecurity_requirements(dataset)
        others = others[others["ID"].isin(folds.loc[folds["fold"] == outer, "ID"])]
        ids, prob, _ = predict_frame(result, model_name, others, labels, config["batch_size"])
        other = prediction_frame(task, ids, None, prob, (prob >= thresholds).astype(int))
        other.insert(1, "outer", outer)
        other.to_csv(fold_dir / NONSECURITY_CSV, index=False, encoding="utf-8-sig")
    del result
    release_memory()

    default_row = trials[trials["is_default_config"]].iloc[0]
    report = {
        "task": task, "outer": outer, "train_real_rows": len(outer_train), "train_synthetic_rows": n_synthetic,
        "test_rows": len(outer_test),
        "optuna": {"trials_finished": len(trials), "trials_pruned": int(trials["state"].eq("PRUNED").sum())},
        "default_config_inner_cv_objective": float(default_row["cv_objective_mean"]),
        "best_trial": {"trial_id": winner_id, "inner_cv_objective_mean": float(winner["cv_objective_mean"]),
                       "inner_cv_objective_std": float(winner["cv_objective_std"])},
        "winning_config": config, "final_training_epochs": final_epochs,
        "thresholds_by_label": dict(zip(labels, thresholds.tolist())),
        "test": {**{name: scores[name] for name in metric_names(task)}, "loss": test_loss},
        "test_per_label": scores["per_label"].to_dict("records"),
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    print(f"  externo {outer + 1} prueba:", {k: round(v, 4) for k, v in report["test"].items()}, flush=True)
    save_json(result_path, report)
    return report


def run(task: str, condition: str, dataset: str, model_name: str, n_trials: int | None = None,
        force: bool = False, save_checkpoints: bool = False) -> dict:
    spec = get_task(task)
    if condition not in spec.conditions:
        raise ValueError(f"La tarea {task} no tiene la condición {condition} (solo {list(spec.conditions)}).")
    labels = list(spec.labels)
    out_dir = run_dir(task, condition, dataset, model_name)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{task}/{condition}/{dataset}/{model_name}: ya completado; use --force para repetirlo.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    n_trials = n_trials or N_TRIALS[model_name]

    # datos, folds y etiquetas evaluadas
    data = load_task_data(dataset, task)
    folds = outer_folds(dataset)
    synthetic = load_synthetic(dataset, data) if condition == "c1_augmented" else None
    active = active_labels(data, task)
    (SPLITS_DIR / task).mkdir(parents=True, exist_ok=True)
    label_coverage(data, task).to_csv(SPLITS_DIR / task / f"{dataset}_label_coverage.csv", index=False,
                                      encoding="utf-8-sig")
    scored = [label for label, keep in zip(labels, active) if keep]
    print(f"\n=== {task} | {condition} | {DATASET_NAMES[dataset]} / {MODEL_DISPLAY[model_name]} | {DEVICE} ===")
    print(f"reales={len(data)} | sintéticos={0 if synthetic is None else len(synthetic)} | "
          f"CV anidada {N_OUTER}x{N_INNER} | trials por externo={n_trials} | etiquetas evaluadas={scored}",
          flush=True)

    outer_reports = [run_outer_fold(task, condition, dataset, model_name, k, data, folds, synthetic, active,
                                    n_trials, out_dir, save_checkpoints) for k in range(N_OUTER)]

    # predicciones de los 5 folds de prueba juntas: cada requisito real aparece una vez
    pooled = pd.concat([pd.read_csv(out_dir / f"outer_{k}" / "outer_test_predictions.csv", encoding="utf-8-sig",
                                    dtype={"ID": str}) for k in range(N_OUTER)], ignore_index=True)
    if sorted(pooled["ID"]) != sorted(data["ID"]):
        raise AssertionError("Las predicciones externas no cubren exactamente una vez todo el dataset.")
    pooled.to_csv(out_dir / "pooled_outer_test_predictions.csv", index=False, encoding="utf-8-sig")

    report = {
        "task": task, "condition": condition, "dataset": dataset, "model": model_name, "seed": SEED,
        "n_outer": N_OUTER, "n_inner": N_INNER, "n_trials_per_outer": n_trials,
        "objective": spec.objective, "tiebreak": spec.tiebreak, "label_cols": labels,
        "rows": len(data), "synthetic_rows": 0 if synthetic is None else len(synthetic),
        "labels_in_macro_average": scored,
        "labels_excluded_from_macro_average": [label for label in labels if label not in scored],
        "label_positives": data[labels].sum().astype(int).to_dict(),
        "search_space": SEARCH_SPACES[model_name], "default_config": DEFAULT_CONFIGS[model_name],
        "outer_folds": outer_reports,
        "elapsed_minutes": round(sum(r["elapsed_minutes"] for r in outer_reports), 1),
    }
    add_pooled_scores(task, report, pooled, active)
    write_run_tables(report, out_dir, task)
    report["pooled_outer_test_confusion"] = save_run_confusion(condition, dataset, model_name, task).to_dict("records")
    for name in metric_names(task):
        summary = report["outer_aggregate"][name]
        print(f"{name}: externos {summary['mean']:.4f} ± {summary['std']:.4f} | agrupado "
              f"{report['pooled_outer_test'][name]:.4f}")
    save_json(summary_path, report)
    update_comparison_table(condition, task)
    export_excel(condition, task)
    return report
