"""Ejecuta secuencialmente los cuatro notebooks de entrenamiento."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
NOTEBOOKS = [
    "01_CNN_multilabel.ipynb",
    "02_BiLSTM_multilabel.ipynb",
    "03_BETO_multilabel.ipynb",
    "04_RoBERTa_BNE_multilabel.ipynb",
]


def stamp(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


for notebook in NOTEBOOKS:
    stamp(f"INICIO {notebook}")
    payload = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(ROOT / notebook),
        "display": lambda value: print(value, flush=True),
    }
    for cell_number, cell in enumerate(payload["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{notebook}:cell-{cell_number}", "exec"), namespace)
        except Exception:
            stamp(f"ERROR {notebook}, celda {cell_number}")
            traceback.print_exc()
            raise
    stamp(f"FIN {notebook}")

stamp("ENTRENAMIENTO COMPLETO")
