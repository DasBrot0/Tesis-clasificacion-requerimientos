#!/usr/bin/env python3
"""Prepara una corrida terminada para extender su búsqueda de Optuna a más trials.

    python scripts/extend_trials.py --condition c1_augmented --models roberta_bne
    python train.py --condition c1_augmented --dataset promise --model roberta_bne --n-trials 12

Conserva los estudios de Optuna (outer_<k>/optuna_study.db y outer_<k>/trials/), así que la corrida
siguiente solo ejecuta los trials que faltan y vuelve a entrenar el modelo de cada fold externo.
Antes de borrar, copia el resultado de la etapa actual a stage_<n>trials/ (summary.json, métricas
agrupadas y outer_<k>/outer_result.json) para que quede registro de cada etapa.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONDITIONS, DATASETS, MODELS, N_OUTER, run_dir  # noqa: E402

RESULT_FILES = ["summary.json", "outer_fold_metrics.csv", "pooled_metrics_by_label.csv",
                "pooled_outer_test_predictions.csv"]
OUTER_FILES = ["outer_result.json", "outer_test_predictions.csv", "optuna_trials.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    for model in args.models:
        for dataset in args.datasets:
            out = run_dir(args.condition, dataset, model)
            summary = out / "summary.json"
            if not summary.exists():
                print(f"{out}: sin summary.json; no hay etapa terminada que extender")
                continue
            n_trials = json.loads(summary.read_text(encoding="utf-8"))["n_trials_per_outer"]
            stage = out / f"stage_{n_trials}trials"
            stage.mkdir(exist_ok=True)
            for name in RESULT_FILES:
                if (out / name).exists():
                    shutil.copy2(out / name, stage / name)
                    (out / name).unlink()
            for outer in range(N_OUTER):
                fold = out / f"outer_{outer}"
                (stage / fold.name).mkdir(exist_ok=True)
                for name in OUTER_FILES:
                    if (fold / name).exists():
                        shutil.copy2(fold / name, stage / fold.name / name)
                        if name != "optuna_trials.csv":
                            (fold / name).unlink()
            print(f"{out}: etapa de {n_trials} trials archivada en {stage.name}; lista para extender")


if __name__ == "__main__":
    main()
