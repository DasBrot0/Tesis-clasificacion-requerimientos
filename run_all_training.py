"""Ejecuta en secuencia todas las corridas del experimento.

    python run_all_training.py                                   # todo: binary C0 + subcategories C0 y C1
    python run_all_training.py --tasks subcategories --conditions c1_augmented
    python run_all_training.py --models cnn bilstm --datasets secreq

Cada corrida escribe su salida en outputs/logs/<tarea>_<condición>_<dataset>_<modelo>.log. Las
corridas terminadas se omiten (salvo --force) y una interrumpida retoma desde el último fold externo
terminado; dentro de ese fold, su estudio de Optuna continúa donde quedó.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import CONDITIONS, DATASETS, LOGS, MODELS, TASKS, log_name

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def stamp(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    extra = (["--n-trials", str(args.n_trials)] if args.n_trials else []) + (["--force"] if args.force else [])
    # Los modelos van en el bucle externo para que terminen primero los rápidos (CNN, BiLSTM).
    runs = [(task, condition, dataset, model)
            for model in args.models for task in args.tasks for condition in args.conditions
            if condition in TASKS[task].conditions for dataset in args.datasets]
    failed = []
    for task, condition, dataset, model in runs:
        name = f"{task}/{condition}/{dataset}/{model}"
        stamp(f"INICIO {name}")
        started = time.time()
        log_path = LOGS / log_name(task, condition, dataset, model)
        with open(log_path, "a", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "train.py"), "--task", task, "--condition", condition,
                 "--dataset", dataset, "--model", model, *extra],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        minutes = (time.time() - started) / 60
        if completed.returncode:
            failed.append(name)
            stamp(f"ERROR {name} ({minutes:.1f} min; ver {log_path.name})")
        else:
            stamp(f"FIN {name} ({minutes:.1f} min)")
    stamp("ENTRENAMIENTO COMPLETO" if not failed else f"TERMINADO CON ERRORES: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
