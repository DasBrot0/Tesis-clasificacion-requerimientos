#!/usr/bin/env python3
"""Corrida provisional: entrena los 12 pares (dataset, modelo) en secuencia y sigue si uno falla.

Uso (desacoplado):  pythonw scripts/run_provisional_training.py <ruta_log>
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [(d, m) for d in ("promise", "secreq", "dosspre") for m in ("cnn", "bilstm", "beto", "roberta_bne")]


def main(log_path: str) -> None:
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        def say(msg):
            log.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")

        say(f"INICIO corrida provisional: {len(PAIRS)} pares")
        for dataset, model in PAIRS:
            say(f"=== {dataset} {model}: inicio")
            t0 = time.time()
            proc = subprocess.run([sys.executable, "train.py", "--dataset", dataset, "--model", model],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            say(f"=== {dataset} {model}: fin código={proc.returncode} ({(time.time() - t0) / 60:.1f} min)")
        say("FIN corrida provisional")


if __name__ == "__main__":
    main(sys.argv[1])
