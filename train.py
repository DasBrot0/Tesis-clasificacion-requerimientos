"""Entrena un modelo sobre un dataset con CV anidada (5 externos x 3 internos + Optuna).

    python train.py --task binary --dataset secreq --model beto
    python train.py --task subcategories --condition c1_augmented --dataset dosspre --model cnn

Tareas: binary (seguridad / no seguridad, todos los requisitos; solo c0_real) y subcategories
(solo requisitos de seguridad, cuatro subcategorías; c0_real y c1_augmented).
"""

from __future__ import annotations

import argparse
import os
import sys

from src.config import CONDITIONS, DATASETS, MODELS, TASKS
from src.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="c0_real",
                        help="c0_real: solo reales; c1_augmented: + sintéticos en entrenamiento (solo subcategories).")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Trials de Optuna por fold externo (por defecto, N_TRIALS de src/config.py).")
    parser.add_argument("--force", action="store_true", help="Repite una corrida ya completada.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Guarda el modelo de cada fold externo.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.task, args.condition, args.dataset, args.model, args.n_trials, args.force, args.save_checkpoints)
    # Una LSTM de varias capas con cuDNN hace caer el intérprete al cerrarse en Windows aunque todo
    # ya esté guardado; se sale sin ese cierre.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
