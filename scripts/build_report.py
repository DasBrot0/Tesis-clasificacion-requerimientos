#!/usr/bin/env python3
"""Regenera los reportes de las corridas terminadas, sin reentrenar.

    python scripts/build_report.py
    python scripts/build_report.py --task subcategories --condition c1_augmented

Por cada (tarea, condición) con corridas terminadas reescribe, a partir de summary.json y de
pooled_outer_test_predictions.csv de cada corrida:
- las métricas por fold y por etiqueta y la matriz de confusión de cada corrida;
- outputs/output_raw/<tarea>/<condición>/: model_comparison.csv, scott_knott_esd.csv, pruebas de
  Cochran y McNemar, confusión entre subcategorías, errores, mejor modelo y Transformers vs. DL;
- outputs/reports/<tarea>/<condición>/results_<tarea>_<condición>.xlsx y las figuras de confusión.
Con las dos tareas terminadas evalúa también la cascada (outputs/output_raw/cascade/<condición>/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cascade import cascade_table  # noqa: E402
from src.config import TASKS  # noqa: E402
from src.reporting import export_excel, finished_runs, rescore_run, update_comparison_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(TASKS), nargs="+", default=list(TASKS))
    parser.add_argument("--condition", nargs="+", default=None)
    args = parser.parse_args()
    for task in args.task:
        for condition in args.condition or TASKS[task].conditions:
            runs = finished_runs(condition, task)
            if not runs:
                print(f"{task}/{condition}: sin corridas terminadas")
                continue
            for dataset, model_name, _ in runs:
                rescore_run(condition, dataset, model_name, task)
            update_comparison_table(condition, task)
            print(f"{task}/{condition}: {len(runs)} corridas -> {export_excel(condition, task)}")
    for condition in args.condition or TASKS["subcategories"].conditions:
        table = cascade_table(condition)
        if not table.empty:
            print(f"cascada/{condition}: {len(table)} pares evaluados")


if __name__ == "__main__":
    main()
