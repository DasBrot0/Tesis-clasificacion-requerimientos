"""End-to-end run for one (dataset, model) pair.

1. Hold out the stratified test and split the rest into five multilabel-stratified folds.
2. Optuna tunes the hyperparameters on the CV mean macro-F1 (trial 0 = default configuration).
3. Per-label thresholds come only from the winner's out-of-fold predictions.
4. The final model is trained once on the complete CV pool, saved, and evaluated on test.
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
    ARTIFACTS, DATASET_NAMES, DATASETS, DEFAULT_CONFIGS, LABEL_COLS, MAX_LENGTH, MODEL_DISPLAY,
    MODELS, N_SPLITS, N_TRIALS, SEARCH_SPACES, SEED, WORD_MODELS, checkpoint_dir, run_dir,
)
from .data import (
    SPLITS_DIR, active_label_mask, get_cv_folds, get_cv_pool_and_test, label_coverage,
    load_dataset, partition_distribution,
)
from .metrics import (
    METRIC_NAMES, aggregate_rows, multilabel_metrics, prediction_frame, save_json,
    select_per_label_thresholds,
)
from .models import DEVICE
from .training import fit, predict, release_memory
from .tuning import trial_paths, trials_frame, tune


def load_protocol(dataset: str):
    """Dataset, CV pool, test, folds and scored labels; identical for the four models."""
    data = load_dataset(dataset)
    cv_pool, test_frame = get_cv_pool_and_test(data, dataset)
    folds = get_cv_folds(cv_pool, dataset)
    label_coverage(data).to_csv(SPLITS_DIR / f"{dataset}_label_coverage.csv", index=False, encoding="utf-8-sig")
    partition_distribution(cv_pool, test_frame, folds).to_csv(
        SPLITS_DIR / f"{dataset}_partition_label_distribution.csv", index=False, encoding="utf-8-sig")
    return data, cv_pool, test_frame, folds, active_label_mask(data)


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
        f1_ax.plot(part["epoch"], part["val_macro_f1"], alpha=0.8, label=f"fold {fold + 1}")
    loss_ax.set(title="BCE loss por fold", xlabel="época", ylabel="loss")
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


def update_comparison_table() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model_name in MODELS:
            path = run_dir(dataset, model_name) / "summary.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            cv, test, default = report["cv_aggregate"], report["final_test"], report["default_config_cv"]
            fmt = lambda name: f"{cv[name]['mean']:.4f} ± {cv[name]['std']:.4f}"
            rows.append({
                "Dataset": DATASET_NAMES[dataset], "Modelo": MODEL_DISPLAY[model_name],
                "CV Macro-F1 (config. por defecto, umbral 0.50)": default["cv_macro_f1_mean"],
                "CV Macro-F1 (Optuna, umbral 0.50)": report["best_trial"]["cv_macro_f1_mean"],
                "CV Macro-F1": fmt("macro_f1"), "CV Weighted-F1": fmt("weighted_f1"),
                "CV Recall Security": fmt("security_recall"),
                "Test Macro-F1": test["macro_f1"], "Test Weighted-F1": test["weighted_f1"],
                "Test Micro-F1": test["micro_f1"], "Test Recall Security": test["security_recall"],
                "Etiquetas en el promedio": ", ".join(report["labels_in_macro_average"]),
            })
    table = pd.DataFrame(rows)
    table.to_csv(ARTIFACTS / "model_comparison.csv", index=False, encoding="utf-8-sig")
    return table


def run(dataset: str, model_name: str, n_trials: int | None = None, force: bool = False) -> dict[str, Any]:
    out_dir = run_dir(dataset, model_name)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{dataset}/{model_name}: ya completado ({summary_path}); use --force para repetirlo.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    started = time.time()
    n_trials = n_trials or N_TRIALS[model_name]
    data, cv_pool, test_frame, folds, active = load_protocol(dataset)
    active_names = [label for label, keep in zip(LABEL_COLS, active) if keep]
    print(f"\n=== {DATASET_NAMES[dataset]} / {MODEL_DISPLAY[model_name]} | device={DEVICE} ===")
    print(f"filas={len(data)} | pool CV={len(cv_pool)} | test={len(test_frame)} | "
          f"etiquetas evaluadas={active_names}", flush=True)

    # 1) Optuna over the 5-fold CV.
    study = tune(dataset, model_name, cv_pool, folds, active, out_dir, n_trials)
    trials = trials_frame(study, model_name)
    trials.to_csv(out_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    winner = trials.iloc[0]
    winner_id = int(winner["trial_id"])
    winning_config = {**DEFAULT_CONFIGS[model_name], **study.trials[winner_id].params}
    default_row = trials[trials["is_default_config"]].iloc[0]

    # 2) OOF predictions of the winner -> per-label thresholds -> CV metrics.
    oof_path, history_path = trial_paths(out_dir, winner_id)
    stored = np.load(oof_path)
    y_true, y_prob, fold_of = stored["y_true"].astype(int), stored["y_prob"], stored["fold"]
    if sorted(stored["ids"].tolist()) != sorted(cv_pool["ID"].tolist()):
        raise AssertionError("Las predicciones OOF no cubren exactamente una vez todo el pool de CV.")
    thresholds, threshold_search = select_per_label_thresholds(y_true, y_prob, active)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    fold_rows, label_rows = [], []
    for fold in range(N_SPLITS):
        mask = fold_of == fold
        metrics = multilabel_metrics(y_true[mask], y_prob[mask], thresholds, active)
        fold_rows.append({"fold": fold, "best_epoch": int(stored["best_epochs"][fold]),
                          **{name: metrics[name] for name in METRIC_NAMES}})
        labels = metrics["per_label"].copy(); labels.insert(0, "fold", fold)
        label_rows.extend(labels.to_dict("records"))
    aggregate = aggregate_rows(fold_rows)
    label_frame = pd.DataFrame(label_rows)
    label_aggregate = (
        label_frame.groupby("label", sort=False)[["precision", "recall", "f1"]]
        .agg(["mean", "std"]).reset_index()
    )
    label_aggregate.columns = [
        "label" if column[0] == "label" else f"{column[0]}_{column[1]}"
        for column in label_aggregate.columns
    ]
    label_aggregate["support_cv_pool"] = y_true.sum(axis=0)
    label_aggregate["in_macro_average"] = active
    for name in METRIC_NAMES:
        print(f"CV {name}: {aggregate[name]['mean']:.4f} ± {aggregate[name]['std']:.4f}")

    oof = prediction_frame(stored["ids"], y_true, y_prob, y_pred)
    oof.insert(1, "fold", fold_of)
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    history.to_csv(out_dir / "cv_history.csv", index=False, encoding="utf-8-sig")
    save_curves(history, out_dir / "cv_curves.png", f"{DATASET_NAMES[dataset]} — {MODEL_DISPLAY[model_name]}")
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    label_frame.to_csv(out_dir / "cv_fold_metrics_by_label.csv", index=False, encoding="utf-8-sig")
    label_aggregate.to_csv(out_dir / "cv_label_metrics_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_search.to_csv(out_dir / "oof_threshold_search.csv", index=False, encoding="utf-8-sig")

    # 3) Final model: whole CV pool, no validation; test is encoded only after training ends.
    final_epochs = max(1, int(round(float(stored["best_epochs"].mean()))))
    print(f"Modelo final: {final_epochs} épocas sobre {len(cv_pool)} filas, config={winning_config}", flush=True)
    result = fit(model_name, winning_config, cv_pool, SEED, stop_epoch=final_epochs, keep_model=True)
    save_checkpoint(model_name, result, {**winning_config, "final_training_epochs": final_epochs},
                    thresholds, active, checkpoint_dir(dataset, model_name))
    test_data = result.encoder.encode(test_frame)
    test_prob, test_loss = predict(result.model, model_name, test_data, result.criterion,
                                   winning_config["batch_size"])
    test_true = test_frame[LABEL_COLS].to_numpy(dtype=int)
    test_metrics = multilabel_metrics(test_true, test_prob, thresholds, active)
    prediction_frame(test_data.ids, test_true, test_prob, test_metrics["y_pred"]).to_csv(
        out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_metrics["per_label"].to_csv(out_dir / "test_metrics_by_label.csv", index=False, encoding="utf-8-sig")
    final_test = {**{name: test_metrics[name] for name in METRIC_NAMES}, "loss": test_loss}
    print("Test:", {name: round(value, 4) for name, value in final_test.items()}, flush=True)
    del result, test_data
    release_memory()

    report: dict[str, Any] = {
        "dataset": dataset, "model": model_name, "seed": SEED, "n_splits": N_SPLITS,
        "rows": len(data), "cv_pool_size": len(cv_pool), "test_size": len(test_frame),
        "labels_in_macro_average": active_names,
        "labels_excluded_from_macro_average": [l for l in LABEL_COLS if l not in active_names],
        "label_positives": data[LABEL_COLS].sum().astype(int).to_dict(),
        "optuna": {"trials_finished": len(trials),
                   "trials_completed": int(trials["state"].eq("COMPLETE").sum()),
                   "trials_pruned": int(trials["state"].eq("PRUNED").sum()),
                   "search_space": SEARCH_SPACES[model_name]},
        "default_config": DEFAULT_CONFIGS[model_name],
        "default_config_cv": {"cv_macro_f1_mean": float(default_row["cv_macro_f1_mean"]),
                              "cv_macro_f1_std": float(default_row["cv_macro_f1_std"])},
        "best_trial": {"trial_id": winner_id, "cv_macro_f1_mean": float(winner["cv_macro_f1_mean"]),
                       "cv_macro_f1_std": float(winner["cv_macro_f1_std"])},
        "winning_config": winning_config, "final_training_epochs": final_epochs,
        "thresholds_by_label": dict(zip(LABEL_COLS, thresholds.tolist())),
        "cv_folds": fold_rows, "cv_aggregate": aggregate,
        "cv_per_label_aggregate": label_aggregate.to_dict("records"),
        "final_test": final_test,
        "final_test_per_label": test_metrics["per_label"].to_dict("records"),
        "checkpoint": checkpoint_dir(dataset, model_name),
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    save_json(out_dir / "best_config.json", {
        "config": winning_config, "final_training_epochs": final_epochs,
        "thresholds_by_label": report["thresholds_by_label"], "selected_from": "OOF only"})
    save_json(summary_path, report)
    update_comparison_table()
    return report
