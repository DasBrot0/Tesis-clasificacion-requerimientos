"""Búsqueda de hiperparámetros con Optuna dentro de un fold externo.

Cada trial entrena el modelo en los 3 folds internos y devuelve la media del objetivo de la tarea
(F1 de seguridad en binary, macro-F1 en subcategories). El trial 0 es siempre la configuración
por defecto. Los trials que van claramente peor que la mediana se podan antes de terminar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import TrialState

from .config import DEFAULT_CONFIGS, DEFAULT_THRESHOLD, N_INNER, SEARCH_SPACES, SEED, get_task
from .data import add_synthetic, split
from .metrics import compute_metrics
from .training import fit, release_memory

FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)


def suggest_config(trial: optuna.Trial, model_name: str) -> dict:
    """Configuración por defecto con los valores que propone Optuna para cada hiperparámetro del espacio."""
    config = dict(DEFAULT_CONFIGS[model_name])
    for name, (kind, *spec) in SEARCH_SPACES[model_name].items():
        if kind == "categorical":
            config[name] = trial.suggest_categorical(name, spec[0])
        else:
            config[name] = trial.suggest_float(name, spec[0], spec[1], log=(kind == "log_float"))
    return config


def default_params(model_name: str) -> dict:
    return {name: DEFAULT_CONFIGS[model_name][name] for name in SEARCH_SPACES[model_name]}


def inner_cv(task: str, model_name: str, config: dict, outer_train: pd.DataFrame, folds: pd.DataFrame,
             active: np.ndarray, synthetic: pd.DataFrame | None = None,
             trial: optuna.Trial | None = None) -> dict:
    """Entrena los folds internos con una configuración y junta sus predicciones de validación."""
    spec = get_task(task)
    ids, truths, probs, histories, best_epochs, objective, tiebreak = [], [], [], [], [], [], []
    for k in range(N_INNER):
        train, validation = split(outer_train, folds, k)
        train = add_synthetic(train, synthetic)
        # la semilla depende solo del fold: todos los trials parten de la misma inicialización
        result = fit(task, model_name, config, train, SEED + k, val_frame=validation, active=active)
        scores = compute_metrics(task, result.val_true, result.val_prob, DEFAULT_THRESHOLD, active)

        ids += result.val_ids
        truths.append(result.val_true)
        probs.append(result.val_prob)
        histories += [{"fold": k, **row} for row in result.history]
        best_epochs.append(result.best_epoch)
        objective.append(scores[spec.objective])
        tiebreak.append(scores[spec.tiebreak])
        print(f"    fold interno {k + 1}/{N_INNER}: {spec.objective}={scores[spec.objective]:.4f} "
              f"mejor época={result.best_epoch} de {len(result.history)}", flush=True)

        if trial is not None:
            trial.report(float(np.mean(objective)), step=k + 1)
            if k + 1 < N_INNER and trial.should_prune():
                raise optuna.TrialPruned()

    return {"ids": np.array(ids), "y_true": np.vstack(truths), "y_prob": np.vstack(probs),
            "history": pd.DataFrame(histories), "best_epochs": np.array(best_epochs),
            "objective": np.array(objective), "tiebreak": np.array(tiebreak)}


def trial_files(out_dir: Path, number: int) -> tuple[Path, Path]:
    """Predicciones fuera de fold e historial por época de un trial."""
    folder = out_dir / "trials"
    return folder / f"trial_{number:03d}_oof.npz", folder / f"trial_{number:03d}_history.csv"


def tune(task: str, dataset: str, model_name: str, outer_train: pd.DataFrame, folds: pd.DataFrame,
         active: np.ndarray, out_dir: Path, n_trials: int,
         synthetic: pd.DataFrame | None = None) -> optuna.Study:
    """Crea (o retoma) el estudio de Optuna de un fold externo y corre los trials que falten."""
    out_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{task}_{dataset}_{model_name}_{out_dir.name}",
        storage=f"sqlite:///{(out_dir / 'optuna_study.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    # Al retomar, la semilla avanza con los trials ya guardados para no repetir configuraciones.
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials),
                                               n_startup_trials=max(4, n_trials // 3))
    # Trial 0 = configuración por defecto (también si una corrida se cortó en medio de ese trial).
    if not any(t.params == default_params(model_name) and t.state in (TrialState.COMPLETE, TrialState.WAITING)
               for t in study.trials):
        study.enqueue_trial(default_params(model_name))

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, model_name)
        print(f"  trial {trial.number}: {config}", flush=True)
        try:
            output = inner_cv(task, model_name, config, outer_train, folds, active, synthetic, trial)
        except torch.OutOfMemoryError:
            release_memory()
            print("    sin memoria de GPU; el trial se descarta.", flush=True)
            raise optuna.TrialPruned()
        oof_path, history_path = trial_files(out_dir, trial.number)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof_path, ids=output["ids"], y_true=output["y_true"], y_prob=output["y_prob"],
                            best_epochs=output["best_epochs"])
        output["history"].to_csv(history_path, index=False, encoding="utf-8-sig")
        trial.set_user_attr("objective_std", float(output["objective"].std(ddof=1)))
        trial.set_user_attr("tiebreak_mean", float(output["tiebreak"].mean()))
        trial.set_user_attr("mean_best_epoch", float(output["best_epochs"].mean()))
        return float(output["objective"].mean())

    remaining = n_trials - sum(t.state in FINISHED for t in study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    return study


def trials_table(study: optuna.Study, model_name: str) -> pd.DataFrame:
    """Una fila por trial terminado, del mejor al peor (objetivo y luego desempate)."""
    rows = []
    for trial in study.trials:
        if trial.state not in FINISHED:
            continue
        rows.append({
            "trial_id": trial.number,
            "state": trial.state.name,
            "is_default_config": trial.params == default_params(model_name),
            "cv_objective_mean": trial.value if trial.state == TrialState.COMPLETE else np.nan,
            "cv_objective_std": trial.user_attrs.get("objective_std", np.nan),
            "cv_tiebreak_mean": trial.user_attrs.get("tiebreak_mean", np.nan),
            "mean_best_epoch": trial.user_attrs.get("mean_best_epoch", np.nan),
            "folds_run": len(trial.intermediate_values),
            **{**DEFAULT_CONFIGS[model_name], **trial.params},
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["cv_objective_mean", "cv_tiebreak_mean"], ascending=False,
                             na_position="last", kind="stable").reset_index(drop=True)
