"""Ejecuta secuencialmente el protocolo CV de los cuatro modelos."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
MODELS = ["cnn", "bilstm", "beto", "roberta_bne"]


def stamp(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


extra_args = sys.argv[1:]
for model in MODELS:
    stamp(f"INICIO CV {model}")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "security_cross_validation.py"), "--model", model, *extra_args],
            cwd=ROOT, check=True,
        )
    except Exception:
        stamp(f"ERROR CV {model}")
        traceback.print_exc()
        raise
    stamp(f"FIN CV {model}")

stamp("ENTRENAMIENTO COMPLETO")
