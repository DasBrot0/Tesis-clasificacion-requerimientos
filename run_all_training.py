"""Ejecuta secuencialmente los entrenamientos de una condición (3 datasets x 4 modelos).

    python run_all_training.py --condition c1_augmented                  # todo
    python run_all_training.py --condition c1_augmented --models cnn roberta_bne
    python run_all_training.py --condition c0_real --datasets secreq --n-trials 5 --force

Cada corrida escribe su salida en outputs/logs/<condición>_<dataset>_<modelo>.log. Las corridas ya
completadas se omiten (salvo --force) y una interrumpida retoma desde el último fold externo
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

from src.config import CONDITIONS, DATASETS, LOGS, MODELS

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def stamp(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    extra = (["--n-trials", str(args.n_trials)] if args.n_trials else []) + (["--force"] if args.force else [])
    failed = []
    # Los modelos van en el bucle externo: primero terminan los rápidos (CNN, BiLSTM).
    for model in args.models:
        for dataset in args.datasets:
            stamp(f"INICIO {args.condition}/{dataset}/{model}")
            started = time.time()
            log_path = LOGS / f"{args.condition}_{dataset}_{model}.log"
            with open(log_path, "a", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "train.py"), "--condition", args.condition,
                     "--dataset", dataset, "--model", model, *extra],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            minutes = (time.time() - started) / 60
            if completed.returncode:
                failed.append(f"{dataset}/{model}")
                stamp(f"ERROR {args.condition}/{dataset}/{model} ({minutes:.1f} min; ver {log_path.name})")
            else:
                stamp(f"FIN {args.condition}/{dataset}/{model} ({minutes:.1f} min)")
    stamp("ENTRENAMIENTO COMPLETO" if not failed else f"TERMINADO CON ERRORES: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
