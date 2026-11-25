#!/usr/bin/env python3
"""Regenera los reportes de las corridas terminadas sin reentrenar.

    python scripts/build_report.py                          # todas las tareas y condiciones con corridas
    python scripts/build_report.py --condition c1_augmented
    python scripts/build_report.py --task binary subcategories

Tareas: binary y subcategories escriben en outputs/output_raw/<tarea>/<condición>/ y
outputs/reports/<tarea>/<condición>/results_<tarea>_<condición>.xlsx; joint (las corridas anteriores,
un modelo de 5 salidas) conserva las rutas de abajo. La regla de coherencia solo existe en joint. Con
las dos tareas nuevas terminadas, también evalúa la cascada binaria -> subcategorías (src/cascade.py)
en outputs/output_raw/cascade/<condición>/.

A partir de lo que cada corrida ya guardó (summary.json y pooled_outer_test_predictions.csv):
- la regla de coherencia (Security = 0 => subcategorías = 0) sobre las predicciones de prueba: las
  predicciones sin corregir quedan en las columnas <etiqueta>_pred_raw y summary.json guarda las
  métricas con y sin corrección y el número de correcciones
- outputs/output_raw/<condición>/<dataset>/<modelo>/pooled_confusion_by_label.csv,
  outer_fold_metrics[_uncorrected].csv y pooled_metrics_by_label[_uncorrected].csv
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
from src.cascade import cascade_table  # noqa: E402
from src.config import CONDITIONS, TASKS, task_spec  # noqa: E402
from src.reporting import export_excel, finished_runs, rescore_run, update_comparison_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, nargs="+", default=CONDITIONS)
    parser.add_argument("--task", choices=TASKS, nargs="+", default=TASKS)
    args = parser.parse_args()
    for task in args.task:
        for condition in args.condition:
            runs = finished_runs(condition, task)
            if not runs:
                print(f"{task}/{condition}: sin corridas terminadas")
                continue
            for dataset, model_name, _ in runs:
                report = rescore_run(condition, dataset, model_name, task)
                if task_spec(task).coherence:
                    fixes = report["coherence_corrections"]
                    print(f"  {dataset}/{model_name}: {fixes['predictions']} predicciones corregidas "
                          f"en {fixes['requirements']} requisitos")
            update_comparison_table(condition, task)
            print(f"{task}/{condition}: {len(runs)} corridas -> {export_excel(condition, task)}")
    if {"binary", "subcategories"} <= set(args.task):
        for condition in args.condition:
            table = cascade_table(condition)
            if not table.empty:
                print(f"cascada/{condition}: {len(table)} pares evaluados")


if __name__ == "__main__":
    main()
