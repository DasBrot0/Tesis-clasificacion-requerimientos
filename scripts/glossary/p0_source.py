#!/usr/bin/env python3
"""P0 del plan de traducción con glosario (ver notas_sesion/PLAN_TRADUCCION_GLOSARIO.md).

Consolida el corpus fuente en inglés de DOSSPRE y SecReq (ePurse-selective, GPS, CPN), con el
origen de cada fila. Las traducciones previas (opus-mt y old/SecReq_es.csv) no se usan: quedaron
como diagnóstico preliminar en old/glossary_p4/.

Salida: datasets/interim/glossary/source_en.csv. La deduplicación se marca sobre el texto en
inglés (columna Dup_group); no se eliminan filas aquí.

Uso:  python scripts/glossary/p0_source.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.prepare_datasets import _dosspre_text, _norm_text  # noqa: E402
from src.config import INTERIM_DIR, ORIGINAL_DIR  # noqa: E402

OUT_DIR = INTERIM_DIR / "glossary"
SECREQ_DIR = ORIGINAL_DIR / "SecReq"
SECREQ_SPECS = ["ePurse-selective", "GPS", "CPN"]

# Los CSV de SecReq mezclan bytes cp1252 sueltos con comillas tipográficas UTF-8 mal convertidas
# ("card issuer\x83??s", "\x83??slot\x83?\x9d"). Se leen como latin1 y se reparan.
_SECREQ_FIXES = [
    (re.compile("\x83\\?\\?([^\x83]*?)\x83\\?\x9d"), r'"\1"'),
    (re.compile("\x83\\?\\?"), "'"),
    (re.compile("\x83\\?\x9d"), '"'),
    (re.compile("\x92"), "'"),
    (re.compile("\x85"), "..."),
    (re.compile("\xa0"), " "),
]


def _secreq_text(value: str) -> str:
    for pattern, repl in _SECREQ_FIXES:
        value = pattern.sub(repl, value)
    return re.sub(r"\s+", " ", value).strip()


def load_secreq() -> pd.DataFrame:
    rows = []
    for spec in SECREQ_SPECS:
        raw = (SECREQ_DIR / f"{spec}.csv").read_bytes().decode("latin1")
        n = 0
        for line in re.split(r"\r?\n", raw):
            if not line.strip() or ";" not in line:
                continue
            text, label = line.rsplit(";", 1)
            label = label.strip().lower()
            if label not in {"sec", "nonsec"}:
                print(f"  secreq/{spec}: se descarta fila con etiqueta {label!r}: {text[:60]!r}")
                continue
            n += 1
            rows.append({"Dataset": "secreq", "Spec": spec, "Source_ID": f"{spec}_{n:03d}",
                         "Requirement_en": _secreq_text(text), "Label": label})
        print(f"  secreq/{spec}: {n} filas")
    frame = pd.DataFrame(rows)
    leftover = frame["Requirement_en"].str.contains("[\x80-\x9f]", regex=True)
    if leftover.any():
        raise ValueError(f"Quedan caracteres de control en SecReq: {frame.loc[leftover, 'Source_ID'].tolist()}")
    return frame


def load_dosspre() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_DIR / "DOSSPRE 1.0Original.csv", encoding="cp1252")
    binary = pd.read_csv(ORIGINAL_DIR / "DOSSPRE 2.0Binary.csv", encoding="cp1252")
    if not original["Project ID"].equals(binary["Project ID"]):
        raise ValueError("DOSSPRE 1.0 y 2.0 no están alineados fila a fila.")
    frame = pd.DataFrame({
        "Dataset": "dosspre",
        "Spec": "DOSSPRE_P" + original["Project ID"].astype(str),
        "Source_ID": [f"dosspre_{i:04d}" for i in range(1, len(original) + 1)],
        "Requirement_en": original["Requirement"].map(_dosspre_text),
        "Label": binary["Class"].str.strip() + "|" + original["Class"].str.strip(),
    })
    print(f"  dosspre: {len(frame)} filas")
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([load_dosspre(), load_secreq()], ignore_index=True)
    key = frame["Dataset"] + "::" + frame["Requirement_en"].map(_norm_text)
    frame["Dup_group"] = key.map({k: i for i, k in enumerate(dict.fromkeys(key))})
    frame["Is_first_occurrence"] = ~key.duplicated()
    out = OUT_DIR / "source_en.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    for name, part in frame.groupby("Dataset"):
        print(f"{name}: {len(part)} filas, {part['Is_first_occurrence'].sum()} textos únicos en inglés")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
