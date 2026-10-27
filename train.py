"""Entrena un modelo sobre un dataset con CV anidada (5 externos x 3 internos + Optuna).

    python train.py --condition c1_augmented --dataset secreq --model roberta_bne
"""

from __future__ import annotations

import argparse
import os
import sys

from src.config import CONDITIONS, DATASETS, MODELS
from src.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, required=True,
                        help="c0_real: solo reales; c1_augmented: reales + sintéticos en entrenamiento.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Trials de Optuna por fold externo (por defecto: N_TRIALS de src/config.py).")
    parser.add_argument("--force", action="store_true", help="Repite una corrida ya completada.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Guarda el modelo de cada fold externo en checkpoints/.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.dataset, args.model, args.n_trials, args.force, args.condition, args.save_checkpoints)
    # A multi-layer cuDNN LSTM crashes the interpreter while it tears down (exit code 127 on
    # Windows) even though everything is already saved; leave without running that teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
