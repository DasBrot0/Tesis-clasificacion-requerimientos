#!/usr/bin/env python3
"""Regenera los reportes de las corridas terminadas sin reentrenar.

    python scripts/build_report.py                          # todas las condiciones con corridas
    python scripts/build_report.py --condition c1_augmented

Por cada corrida (artifacts/<condición>/<dataset>/<modelo>/) escribe la matriz de confusión por
clase (pooled_confusion_by_label.csv y pooled_confusion_matrices.png) a partir de
pooled_outer_test_predictions.csv. Por condición, actualiza model_comparison.csv y
results_<condición>.xlsx. train.py hace lo mismo al terminar cada corrida.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONDITIONS, DATASET_NAMES, MODEL_DISPLAY, run_dir  # noqa: E402
from src.reporting import export_excel, finished_runs, save_run_confusion, update_comparison_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, nargs="+", default=CONDITIONS)
    args = parser.parse_args()
    for condition in args.condition:
        runs = finished_runs(condition)
        if not runs:
            print(f"{condition}: sin corridas terminadas")
            continue
        for dataset, model_name, _ in runs:
            save_run_confusion(run_dir(condition, dataset, model_name),
                               f"{DATASET_NAMES[dataset]} — {MODEL_DISPLAY[model_name]} ({condition})")
        update_comparison_table(condition)
        print(f"{condition}: {len(runs)} corridas -> {export_excel(condition)}")


if __name__ == "__main__":
    main()
