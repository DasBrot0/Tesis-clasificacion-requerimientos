#!/usr/bin/env python3
"""P6: traducción principal con DeepL + glosario (REST, sin SDK).

    glossary   -> crea el glosario EN->ES desde glossary_en_es_provisional.tsv (guarda su id)
    translate  -> traduce los textos únicos en inglés de source_en.csv -> translations/deepl_provisional.jsonl
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "datasets" / "interim" / "glossary"
T = ROOT / "datasets" / "interim" / "translations"
API = "https://api-free.deepl.com"
CONTEXT = "Software security requirements specification (ISO/IEC 25010 security)."
TSV = G / "glossary_en_es_provisional.tsv"
META = T / "deepl_glossary_provisional.json"
OUT = T / "deepl_provisional.jsonl"
RAW = T / "deepl_raw"


def key() -> str:
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("DEEPL_AUTH_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("falta DEEPL_AUTH_KEY")


H = {"Authorization": f"DeepL-Auth-Key {key()}"}


def usage():
    return requests.get(f"{API}/v2/usage", headers=H, timeout=30).json()


def glossary():
    df = pd.read_csv(TSV, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
    df["en"] = df["en"].str.strip()
    df["es"] = df["es"].str.strip()
    assert df["en"].is_unique and (df["es"] != "").all() and (df["en"] != "").all()
    entries = "\n".join(f"{a}\t{b}" for a, b in zip(df.en, df.es))
    h = hashlib.sha256(entries.encode("utf-8")).hexdigest()[:12]
    r = requests.post(f"{API}/v2/glossaries", headers=H, timeout=60, json={
        "name": f"seminario-secreq-dosspre-provisional-{h}", "source_lang": "EN", "target_lang": "ES",
        "entries": entries, "entries_format": "tsv"})
    print(r.status_code, r.text[:500])
    r.raise_for_status()
    gl = r.json()
    META.write_text(json.dumps({"glossary_id": gl["glossary_id"], "entries_sha256_12": h, "n_entries": len(df),
                                "response": gl, "created_utc": datetime.now(timezone.utc).isoformat()},
                               indent=2, ensure_ascii=False), encoding="utf-8")


def translate():
    meta = json.loads(META.read_text(encoding="utf-8"))
    gid, gh = meta["glossary_id"], meta["entries_sha256_12"]
    src = pd.read_csv(G / "source_en.csv", encoding="utf-8-sig")
    grp = src.groupby("Requirement_en", sort=False).agg(
        Dup_group=("Dup_group", "first"), Source_IDs=("Source_ID", list), Datasets=("Dataset", "first")).reset_index()
    done = {}
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            done[r["en"]] = r
    todo = [r for r in grp.itertuples() if r.Requirement_en not in done]
    nchars = sum(len(r.Requirement_en) for r in todo)
    u = usage()
    print(f"{len(grp)} textos únicos; pendientes {len(todo)} ({nchars} caracteres); uso {u}")
    if u["character_limit"] - u["character_count"] < nchars * 1.05:
        raise SystemExit("cuota insuficiente")
    RAW.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        body = {"text": [c.Requirement_en for c in chunk], "source_lang": "EN", "target_lang": "ES",
                "glossary_id": gid, "model_type": "quality_optimized", "split_sentences": "nonewlines",
                "preserve_formatting": True, "context": CONTEXT, "show_billed_characters": True}
        for attempt in range(8):
            r = requests.post(f"{API}/v2/translate", headers=H, json=body, timeout=120)
            if r.status_code == 456:
                raise SystemExit("456: cuota agotada")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise SystemExit("reintentos agotados")
        resp = r.json()
        (RAW / f"batch_{i // 50:03d}.json").write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
        ts = datetime.now(timezone.utc).isoformat()
        with open(OUT, "a", encoding="utf-8") as fh:
            for c, t in zip(chunk, resp["translations"]):
                fh.write(json.dumps({"en": c.Requirement_en, "Dup_group": int(c.Dup_group),
                                     "Source_IDs": c.Source_IDs, "es": t["text"],
                                     "model_type_used": t.get("model_type_used"), "glossary_id": gid,
                                     "glossary_hash": gh, "timestamp_utc": ts, "raw": t},
                                    ensure_ascii=False) + "\n")
        print(f"  lote {i // 50 + 1}/{(len(todo) + 49) // 50}", flush=True)
    print("uso final", usage())


if __name__ == "__main__":
    {"glossary": glossary, "translate": translate, "usage": lambda: print(usage())}[sys.argv[1]]()
