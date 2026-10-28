"""End-to-end nested-CV run for one (condition, dataset, model) triple.

For each of the N_OUTER outer folds:
1. Optuna tunes the hyperparameters on the N_INNER inner folds of the outer training part
   (trial 0 = default configuration).
2. Per-label thresholds come only from the winner's inner out-of-fold predictions.
3. The model is retrained on the whole outer training part and predicts the outer test fold once.

The outer test folds cover every real requirement exactly once; their pooled predictions give the
reported metrics. In c1_augmented the synthetic rows join the training parts (inner and outer)
only through their anchors; no synthetic row is ever validated or tested.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import (
    CONDITIONS, DATASET_NAMES, DEFAULT_CONFIGS, LABEL_COLS, MAX_LENGTH, MODEL_DISPLAY, N_INNER,
    N_OUTER, N_TRIALS, SEARCH_SPACES, SEED, WORD_MODELS, checkpoint_dir, run_dir,
)
from .data import (
    SPLITS_DIR, active_label_mask, get_inner_folds, get_outer_folds, label_coverage, load_dataset,
    load_synthetic, partition_distribution, split_by_fold, with_synthetic,
)
from .metrics import (
    METRIC_NAMES, aggregate_rows, multilabel_metrics, prediction_frame, save_json,
    select_per_label_thresholds,
)
from .models import DEVICE
from .reporting import export_excel, save_run_confusion, update_comparison_table
from .training import fit, predict, release_memory
from .tuning import trial_paths, trials_frame, tune


def load_protocol(condition: str, dataset: str):
    """Dataset, outer folds, scored labels and (for c1_augmented) the synthetic rows."""
    if condition not in CONDITIONS:
        raise ValueError(f"Condición desconocida: {condition}")
    data = load_dataset(dataset)
    outer_folds = get_outer_folds(data, dataset)
    synthetic = load_synthetic(dataset, data) if condition == "c1_augmented" else None
    label_coverage(data).to_csv(SPLITS_DIR / f"{dataset}_label_coverage.csv", index=False, encoding="utf-8-sig")
    partition_distribution(data, dataset, synthetic).to_csv(
        SPLITS_DIR / f"{dataset}_{condition}_partition_label_distribution.csv", index=False, encoding="utf-8-sig")
    return data, outer_folds, synthetic, active_label_mask(data)


def save_curves(history: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for fold, part in history.groupby("fold"):
        loss_ax.plot(part["epoch"], part["train_loss"], color="tab:blue", alpha=0.6,
                     label="train" if fold == 0 else None)
        loss_ax.plot(part["epoch"], part["val_loss"], color="tab:orange", alpha=0.6,
                     label="validation" if fold == 0 else None)
        f1_ax.plot(part["epoch"], part["val_macro_f1"], alpha=0.8, label=f"fold interno {fold + 1}")
    loss_ax.set(title="BCE loss por fold interno", xlabel="época", ylabel="loss")
    f1_ax.set(title="Macro-F1 de validación (umbral 0.50)", xlabel="época", ylabel="macro-F1")
    loss_ax.legend(); f1_ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_checkpoint(model_name: str, result, config: dict, thresholds: np.ndarray,
                    active: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"config": config, "thresholds_by_label": dict(zip(LABEL_COLS, thresholds.tolist())),
            "label_cols": LABEL_COLS, "labels_in_macro_average": dict(zip(LABEL_COLS, active.tolist())),
            "seed": SEED}
    if model_name in WORD_MODELS:
        torch.save({"model_state_dict": result.model.state_dict(), "vocab": result.encoder.vocab,
                    "max_len": result.encoder.max_len, **meta}, directory / "model.pt")
    else:
        result.model.save_pretrained(directory)
        result.encoder.tokenizer.save_pretrained(directory)
        meta.update({"model_source_used": result.encoder.source, "max_length": MAX_LENGTH})
    save_json(directory / "metrics_config.json", meta)


def run_outer_fold(condition: str, dataset: str, model_name: str, outer: int, data: pd.DataFrame,
                   outer_folds: pd.DataFrame, synthetic: pd.DataFrame | None, active: np.ndarray,
                   n_trials: int, out_dir: Path, save_checkpoints: bool) -> dict[str, Any]:
    fold_dir = out_dir / f"outer_{outer}"
    result_path = fold_dir / "outer_result.json"
    if result_path.exists():
        print(f"  externo {outer + 1}/{N_OUTER}: ya completado", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    started = time.time()
    outer_train, outer_test = split_by_fold(data, outer_folds, outer)
    inner_folds = get_inner_folds(outer_train, dataset, outer)
    print(f"\n--- externo {outer + 1}/{N_OUTER}: entrenamiento={len(outer_train)} reales"
          f"{'' if synthetic is None else f' + {len(with_synthetic(outer_train, synthetic)) - len(outer_train)} sintéticos'}"
          f" | prueba={len(outer_test)} ---", flush=True)

    # 1) Optuna over the inner CV.
    study = tune(dataset, model_name, outer_train, inner_folds, active, fold_dir, n_trials, synthetic)
    trials = trials_frame(study, model_name)
    trials.to_csv(fold_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    winner = trials.iloc[0]
    winner_id = int(winner["trial_id"])
    config = {**DEFAULT_CONFIGS[model_name], **study.trials[winner_id].params}
    default_row = trials[trials["is_default_config"]].iloc[0]

    # 2) Inner OOF of the winner -> per-label thresholds.
    oof_path, history_path = trial_paths(fold_dir, winner_id)
    stored = np.load(oof_path)
    if sorted(stored["ids"].tolist()) != sorted(outer_train["ID"].tolist()):
        raise AssertionError("El OOF interno no cubre exactamente una vez el entrenamiento externo.")
    thresholds, threshold_search = select_per_label_thresholds(
        stored["y_true"].astype(int), stored["y_prob"], active)
    threshold_search.to_csv(fold_dir / "inner_oof_threshold_search.csv", index=False, encoding="utf-8-sig")
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    history.to_csv(fold_dir / "inner_cv_history.csv", index=False, encoding="utf-8-sig")
    save_curves(history, fold_dir / "inner_cv_curves.png",
                f"{DATASET_NAMES[dataset]} — {MODEL_DISPLAY[model_name]} — externo {outer + 1} ({condition})")

    # 3) Retrain on the whole outer training part; the test fold is encoded only afterwards.
    final_epochs = max(1, int(round(float(stored["best_epochs"].mean()))))
    train_frame = with_synthetic(outer_train, synthetic)
    print(f"  modelo del externo {outer + 1}: {final_epochs} épocas sobre {len(train_frame)} filas, "
          f"config={config}", flush=True)
    result = fit(model_name, config, train_frame, SEED + 100 + outer, stop_epoch=final_epochs, keep_model=True)
    if save_checkpoints:
        save_checkpoint(model_name, result, {**config, "final_training_epochs": final_epochs},
                        thresholds, active, checkpoint_dir(condition, dataset, model_name, outer))
    test_data = result.encoder.encode(outer_test)
    test_prob, test_loss = predict(result.model, model_name, test_data, result.criterion, config["batch_size"])
    test_true = outer_test[LABEL_COLS].to_numpy(dtype=int)
    metrics = multilabel_metrics(test_true, test_prob, thresholds, active)
    predictions = prediction_frame(test_data.ids, test_true, test_prob, metrics["y_pred"])
    predictions.insert(1, "outer", outer)
    predictions.to_csv(fold_dir / "outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    del result, test_data
    release_memory()

    report = {
        "outer": outer, "train_real_rows": len(outer_train), "train_synthetic_rows": len(train_frame) - len(outer_train),
        "test_rows": len(outer_test),
        "optuna": {"trials_finished": len(trials), "trials_completed": int(trials["state"].eq("COMPLETE").sum()),
                   "trials_pruned": int(trials["state"].eq("PRUNED").sum())},
        "default_config_inner_cv_macro_f1": float(default_row["cv_macro_f1_mean"]),
        "best_trial": {"trial_id": winner_id, "inner_cv_macro_f1_mean": float(winner["cv_macro_f1_mean"]),
                       "inner_cv_macro_f1_std": float(winner["cv_macro_f1_std"])},
        "winning_config": config, "final_training_epochs": final_epochs,
        "thresholds_by_label": dict(zip(LABEL_COLS, thresholds.tolist())),
        "test": {**{name: metrics[name] for name in METRIC_NAMES}, "loss": test_loss},
        "test_per_label": metrics["per_label"].to_dict("records"),
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    print(f"  externo {outer + 1} prueba:", {k: round(v, 4) for k, v in report["test"].items()}, flush=True)
    save_json(result_path, report)
    return report


def run(dataset: str, model_name: str, n_trials: int | None = None, force: bool = False,
        condition: str = "c0_real", save_checkpoints: bool = False) -> dict[str, Any]:
    out_dir = run_dir(condition, dataset, model_name)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{condition}/{dataset}/{model_name}: ya completado ({summary_path}); use --force para repetirlo.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if force and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    n_trials = n_trials or N_TRIALS[model_name]
    data, outer_folds, synthetic, active = load_protocol(condition, dataset)
    active_names = [label for label, keep in zip(LABEL_COLS, active) if keep]
    print(f"\n=== {condition} | {DATASET_NAMES[dataset]} / {MODEL_DISPLAY[model_name]} | device={DEVICE} ===")
    print(f"filas reales={len(data)} | sintéticos={0 if synthetic is None else len(synthetic)} | "
          f"CV anidada {N_OUTER}x{N_INNER} | trials por externo={n_trials} | etiquetas evaluadas={active_names}",
          flush=True)

    outer_reports = [run_outer_fold(condition, dataset, model_name, outer, data, outer_folds, synthetic,
                                    active, n_trials, out_dir, save_checkpoints) for outer in range(N_OUTER)]

    # Pool the five outer test folds: every real requirement is predicted exactly once, each with
    # the thresholds of its own outer fold.
    pooled = pd.concat([pd.read_csv(out_dir / f"outer_{k}" / "outer_test_predictions.csv",
                                    encoding="utf-8-sig", dtype={"ID": str}) for k in range(N_OUTER)],
                       ignore_index=True)
    if sorted(pooled["ID"]) != sorted(data["ID"]):
        raise AssertionError("Las predicciones externas no cubren exactamente una vez todo el dataset.")
    pooled.to_csv(out_dir / "pooled_outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    y_true = pooled[[f"{l}_true" for l in LABEL_COLS]].to_numpy(dtype=int)
    y_pred = pooled[[f"{l}_pred" for l in LABEL_COLS]].to_numpy(dtype=int)
    pooled_metrics = multilabel_metrics(y_true, y_pred.astype(float), 0.5, active)
    fold_rows = [{"outer": r["outer"], **{name: r["test"][name] for name in METRIC_NAMES}} for r in outer_reports]
    aggregate = aggregate_rows(fold_rows)
    pd.DataFrame(fold_rows).to_csv(out_dir / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pooled_metrics["per_label"].to_csv(out_dir / "pooled_metrics_by_label.csv", index=False, encoding="utf-8-sig")
    confusion = save_run_confusion(condition, dataset, model_name)
    for name in METRIC_NAMES:
        print(f"externos {name}: {aggregate[name]['mean']:.4f} ± {aggregate[name]['std']:.4f} | "
              f"agrupado {pooled_metrics[name]:.4f}")

    report: dict[str, Any] = {
        "condition": condition, "dataset": dataset, "model": model_name, "seed": SEED,
        "n_outer": N_OUTER, "n_inner": N_INNER, "n_trials_per_outer": n_trials,
        "rows": len(data), "synthetic_rows": 0 if synthetic is None else len(synthetic),
        "labels_in_macro_average": active_names,
        "labels_excluded_from_macro_average": [l for l in LABEL_COLS if l not in active_names],
        "label_positives": data[LABEL_COLS].sum().astype(int).to_dict(),
        "search_space": SEARCH_SPACES[model_name], "default_config": DEFAULT_CONFIGS[model_name],
        "outer_folds": outer_reports, "outer_aggregate": aggregate,
        "pooled_outer_test": {name: pooled_metrics[name] for name in METRIC_NAMES},
        "pooled_outer_test_per_label": pooled_metrics["per_label"].to_dict("records"),
        "pooled_outer_test_confusion": confusion.to_dict("records"),
        # Sum of the outer folds, so a resumed run still reports its full training time.
        "elapsed_minutes": round(sum(r["elapsed_minutes"] for r in outer_reports), 1),
    }
    save_json(summary_path, report)
    update_comparison_table(condition)
    export_excel(condition)
    return report
