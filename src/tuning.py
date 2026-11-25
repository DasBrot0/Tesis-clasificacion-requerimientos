"""Optuna hyperparameter search whose objective is the mean of the task's objective metric over
the inner CV (macro-F1 in the joint and subcategory tasks, F1 of Security in the binary task).

It runs once per outer fold, on that fold's training part only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import TrialState

from .config import DEFAULT_CONFIGS, DEFAULT_THRESHOLD, N_INNER, SEARCH_SPACES, SEED, task_spec
from .data import split_by_fold, with_synthetic
from .metrics import multilabel_metrics
from .training import fit, release_memory

FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)


def suggest_config(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIGS[model_name])
    for name, (kind, *spec) in SEARCH_SPACES[model_name].items():
        if kind == "categorical":
            config[name] = trial.suggest_categorical(name, spec[0])
        else:
            config[name] = trial.suggest_float(name, spec[0], spec[1], log=kind == "log_float")
    return config


def trial_paths(out_dir: Path, number: int) -> tuple[Path, Path]:
    trials_dir = out_dir / "trials"
    return trials_dir / f"trial_{number:03d}_oof.npz", trials_dir / f"trial_{number:03d}_history.csv"


def run_cv(model_name: str, config: dict, cv_pool: pd.DataFrame, folds: pd.DataFrame,
           active: np.ndarray, trial: optuna.Trial | None = None,
           synthetic: pd.DataFrame | None = None, task: str = "joint") -> dict[str, Any]:
    """Train the inner folds; report the running mean to Optuna so weak trials are pruned early.

    Synthetic rows join each training fold only through their anchor; validation stays real.
    """
    spec = task_spec(task)
    ids, fold_ids, truths, probs, histories, best_epochs, fold_scores, recalls = [], [], [], [], [], [], [], []
    for fold in range(N_INNER):
        train_frame, validation_frame = split_by_fold(cv_pool, folds, fold)
        train_frame = with_synthetic(train_frame, synthetic)
        # The seed depends only on the fold, so every trial starts from the same initialization.
        result = fit(model_name, config, train_frame, SEED + fold, validation_frame, active, task=task)
        metrics = multilabel_metrics(result.val_true, result.val_prob, DEFAULT_THRESHOLD, active, task)
        ids.extend(result.val_ids)
        fold_ids.extend([fold] * len(result.val_ids))
        truths.append(result.val_true)
        probs.append(result.val_prob)
        histories.extend({"fold": fold, **row} for row in result.history)
        best_epochs.append(result.best_epoch)
        fold_scores.append(metrics[spec.objective])
        recalls.append(metrics[spec.tiebreak])
        print(f"    fold={fold + 1}/{N_INNER} {spec.objective}={metrics[spec.objective]:.4f} "
              f"{spec.tiebreak}={metrics[spec.tiebreak]:.4f} best_epoch={result.best_epoch}", flush=True)
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=fold + 1)
            if fold + 1 < N_INNER and trial.should_prune():
                raise optuna.TrialPruned()
    return {"ids": np.array(ids), "fold": np.array(fold_ids), "y_true": np.vstack(truths),
            "y_prob": np.vstack(probs), "history": pd.DataFrame(histories),
            "best_epochs": np.array(best_epochs), "fold_objective": np.array(fold_scores),
            "fold_tiebreak": np.array(recalls)}


def tune(dataset: str, model_name: str, cv_pool: pd.DataFrame, folds: pd.DataFrame,
         active: np.ndarray, out_dir: Path, n_trials: int,
         synthetic: pd.DataFrame | None = None, task: str = "joint") -> optuna.Study:
    """Run (or resume) the study of one outer fold. Trial 0 is always the default configuration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{dataset}_{model_name}_{out_dir.name}",
        storage=f"sqlite:///{(out_dir / 'optuna_study.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    # The seed moves with the trials already stored: a resumed study with the same seed would
    # propose the same startup configurations again.
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials),
                                               n_startup_trials=max(4, n_trials // 3))
    default_params = {name: DEFAULT_CONFIGS[model_name][name] for name in SEARCH_SPACES[model_name]}
    if not any(t.params == default_params and t.state in (TrialState.COMPLETE, TrialState.WAITING)
               for t in study.trials):
        study.enqueue_trial(default_params)

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, model_name)
        print(f"  trial {trial.number}: {config}", flush=True)
        try:
            output = run_cv(model_name, config, cv_pool, folds, active, trial, synthetic, task)
        except torch.OutOfMemoryError:
            release_memory()
            print("    sin memoria de GPU; el trial se descarta.", flush=True)
            raise optuna.TrialPruned()
        oof_path, history_path = trial_paths(out_dir, trial.number)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof_path, ids=output["ids"], fold=output["fold"], y_true=output["y_true"],
                            y_prob=output["y_prob"], best_epochs=output["best_epochs"])
        output["history"].to_csv(history_path, index=False, encoding="utf-8-sig")
        _, std_col, tie_col = trial_columns(task)
        trial.set_user_attr(std_col, float(output["fold_objective"].std(ddof=1)))
        trial.set_user_attr(tie_col, float(output["fold_tiebreak"].mean()))
        trial.set_user_attr("mean_best_epoch", float(output["best_epochs"].mean()))
        return float(output["fold_objective"].mean())

    remaining = n_trials - sum(t.state in FINISHED for t in study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    return study


def trial_columns(task: str = "joint") -> tuple[str, str, str]:
    """(objective mean, objective std, tie-break mean) columns of trials_frame; in the joint task
    cv_macro_f1_mean, cv_macro_f1_std and cv_security_recall_mean, as in the earlier runs."""
    spec = task_spec(task)
    return f"cv_{spec.objective}_mean", f"cv_{spec.objective}_std", f"cv_{spec.tiebreak}_mean"


def trials_frame(study: optuna.Study, model_name: str, task: str = "joint") -> pd.DataFrame:
    """One row per finished trial, best first (the task's objective, then its tie-break)."""
    mean_col, std_col, tie_col = trial_columns(task)
    default_params = {name: DEFAULT_CONFIGS[model_name][name] for name in SEARCH_SPACES[model_name]}
    rows = []
    for trial in study.trials:
        if trial.state not in FINISHED:
            continue
        rows.append({
            "trial_id": trial.number, "state": trial.state.name,
            "is_default_config": trial.params == default_params,
            mean_col: trial.value if trial.state == TrialState.COMPLETE else np.nan,
            std_col: trial.user_attrs.get(std_col, np.nan),
            tie_col: trial.user_attrs.get(tie_col, np.nan),
            "mean_best_epoch": trial.user_attrs.get("mean_best_epoch", np.nan),
            "folds_run": len(trial.intermediate_values),
            **{**DEFAULT_CONFIGS[model_name], **trial.params},
        })
    return pd.DataFrame(rows).sort_values(
        [mean_col, tie_col], ascending=False, kind="stable",
        na_position="last").reset_index(drop=True)
