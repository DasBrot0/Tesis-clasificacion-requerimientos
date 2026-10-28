#!/usr/bin/env python3
"""Regenera los reportes de las corridas terminadas sin reentrenar.

    python scripts/build_report.py                          # todas las condiciones con corridas
    python scripts/build_report.py --condition c1_augmented

A partir de lo que cada corrida ya guardó (summary.json y pooled_outer_test_predictions.csv):
- outputs/output_raw/<condición>/<dataset>/<modelo>/pooled_confusion_by_label.csv
- outputs/output_raw/<condición>/model_comparison.csv y scott_knott_esd.csv
- outputs/output_raw/<condición>/model_tests_cochran_q.csv y model_tests_mcnemar.csv
- outputs/output_raw/<condición>/subcategory_confusion.csv y misclassified.csv
- outputs/output_raw/<condición>/best_model.csv, family_comparison.csv y family_comparison_by_label.csv
- outputs/output_raw/synthetic_diversity.csv (solo con c1_augmented)
- outputs/reports/<condición>/confusion_matrices/<dataset>_<modelo>.png
- outputs/reports/<condición>/results_<condición>.xlsx
train.py hace lo mismo al terminar cada corrida.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONDITIONS  # noqa: E402
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
            save_run_confusion(condition, dataset, model_name)
        update_comparison_table(condition)
        print(f"{condition}: {len(runs)} corridas -> {export_excel(condition)}")


if __name__ == "__main__":
    main()
