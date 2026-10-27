#!/usr/bin/env python3
"""G4: traduce los sintéticos con DeepL + el glosario del plan de traducción (mismos parámetros
que P6) y calcula las señales de calidad de P8 (exactitud terminológica, CometKiwi, LaBSE).

    python scripts/augmentation/g4_translate.py translate
    python scripts/augmentation/g4_translate.py qc

Salidas en datasets/interim/augmentation/: translations.jsonl (respuesta cruda incluida),
qc_signals.csv (en, es, term_acc, cometkiwi, labse).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "glossary"))
import p6_deepl  # noqa: E402  (clave, glosario y parámetros de P6)
import p8_qc  # noqa: E402

A = ROOT / "datasets" / "interim" / "augmentation"
OUT = A / "translations.jsonl"


def translate() -> None:
    meta = json.loads(p6_deepl.META.read_text(encoding="utf-8"))
    gid, gh = meta["glossary_id"], meta["entries_sha256_12"]
    gen = pd.read_csv(A / "generated_raw.csv", encoding="utf-8-sig")
    done = {json.loads(l)["gen_id"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    todo = gen[~gen["gen_id"].isin(done)]
    u = p6_deepl.usage()
    print(f"pendientes {len(todo)} ({todo.requirement_en.str.len().sum()} caracteres); uso {u}")
    if u["character_limit"] - u["character_count"] < todo.requirement_en.str.len().sum() * 1.05:
        raise SystemExit("cuota insuficiente")
    for i in range(0, len(todo), 50):
        chunk = todo.iloc[i:i + 50]
        body = {"text": chunk.requirement_en.tolist(), "source_lang": "EN", "target_lang": "ES",
                "glossary_id": gid, "model_type": "quality_optimized", "split_sentences": "nonewlines",
                "preserve_formatting": True, "context": p6_deepl.CONTEXT, "show_billed_characters": True}
        for attempt in range(8):
            r = requests.post(f"{p6_deepl.API}/v2/translate", headers=p6_deepl.H, json=body, timeout=120)
            if r.status_code == 456:
                raise SystemExit("456: cuota agotada")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise SystemExit("reintentos agotados")
        ts = datetime.now(timezone.utc).isoformat()
        with open(OUT, "a", encoding="utf-8") as fh:
            for row, t in zip(chunk.itertuples(), r.json()["translations"]):
                fh.write(json.dumps({"gen_id": row.gen_id, "en": row.requirement_en, "es": t["text"],
                                     "model_type_used": t.get("model_type_used"), "glossary_id": gid,
                                     "glossary_hash": gh, "timestamp_utc": ts, "raw": t},
                                    ensure_ascii=False) + "\n")
        print(f"  lote {i // 50 + 1}/{(len(todo) + 49) // 50}", flush=True)
    print("uso final", p6_deepl.usage())


def qc() -> None:
    """Reutiliza P8 apuntando sus rutas a los sintéticos."""
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    df = pd.DataFrame(rows)
    tmp = A / "_qc"
    tmp.mkdir(exist_ok=True)
    p8_qc.T = tmp
    p8_qc.load = lambda: df[["en", "es"]].copy()
    p8_qc.terms()  # escribe _qc/_qc_terms.csv y _qc/_qc_neural_in.csv
    subprocess.run([p8_qc.VENV, str(ROOT / "scripts" / "glossary" / "p8_neural_signals.py"),
                    str(tmp / "_qc_neural_in.csv"), str(tmp / "_qc_neural.csv")], check=True)
    terms = pd.read_csv(tmp / "_qc_terms.csv", encoding="utf-8-sig")
    neural = pd.read_csv(tmp / "_qc_neural.csv", encoding="utf-8-sig")
    out = (df[["gen_id", "en", "es"]].merge(terms[["en", "term_acc", "term_missing"]], on="en", how="left")
           .merge(neural[["en", "cometkiwi", "labse"]], on="en", how="left").drop_duplicates("gen_id"))
    out.to_csv(A / "qc_signals.csv", index=False, encoding="utf-8-sig")
    print(out[["term_acc", "cometkiwi", "labse"]].describe())


if __name__ == "__main__":
    {"translate": translate, "qc": qc}[sys.argv[1]]()
