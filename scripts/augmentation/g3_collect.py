#!/usr/bin/env python3
"""G3: valida y consolida las salidas de SON-G (gen_runs/) en generated_raw.csv.

    python scripts/augmentation/g3_collect.py

Comprueba por lote: una línea por dominio pedido, en orden, con anchor_id y domain correctos.
Marca (no elimina) los que no usan verbo modal o reutilizan 4 palabras seguidas de la fuente;
G5 decide.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import AUGMENTATION_DIR  # noqa: E402

A = AUGMENTATION_DIR
MODEL = "claude-sonnet-5-5"
PROMPT = A / "prompt_gen_v1.md"


def _jl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def ngrams(text: str, n: int = 4) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main() -> None:
    sources = pd.read_csv(A / "sources.csv", encoding="utf-8-sig").set_index("anchor_id")
    rows, problems = [], []
    for batch in sorted((A / "gen_batches").glob("batch_*.jsonl")):
        expected = _jl(batch)
        out = A / "gen_runs" / batch.name
        if not out.exists():
            problems.append(f"{batch.name}: falta la salida")
            continue
        got = _jl(out)
        want = [(e["anchor_id"], i + 1, d) for e in expected for i, d in enumerate(e["domains"])]
        have = [(g.get("anchor_id"), g.get("variant"), g.get("domain")) for g in got]
        if want != have or any(set(g) != {"anchor_id", "variant", "domain", "requirement"} for g in got):
            problems.append(f"{batch.name}: {len(have)} líneas, se esperaban {len(want)} con el mismo orden")
            continue
        for g in got:
            src = sources.loc[g["anchor_id"]]
            text = " ".join(str(g["requirement"]).split())
            rows.append({
                "dataset": src["dataset"], "anchor_id": g["anchor_id"], "variant": g["variant"],
                "domain": g["domain"], "labels": src["labels"], "requirement_en": text,
                "has_modal": bool(re.search(r"\b(shall|must)\b", text, re.I)),
                "reuses_4gram": bool(ngrams(text) & ngrams(src["text_en"])),
                "batch": batch.name,
            })
    if problems:
        print("\n".join(problems))
        raise SystemExit("hay lotes por repetir")
    df = pd.DataFrame(rows)
    df.insert(0, "gen_id", [f"g{i:04d}" for i in range(1, len(df) + 1)])
    df.to_csv(A / "generated_raw.csv", index=False, encoding="utf-8-sig")
    meta = {"model": MODEL, "prompt": PROMPT.name,
            "prompt_sha256_12": hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:12],
            "n_batches": len(list((A / "gen_batches").glob("batch_*.jsonl"))), "n_generated": len(df),
            "by_dataset": df["dataset"].value_counts().to_dict(),
            "without_modal": int((~df.has_modal).sum()), "reusing_4gram": int(df.reuses_4gram.sum()),
            "collected_utc": datetime.now(timezone.utc).isoformat(),
            "runs": "un subagente nuevo (claude-sonnet-5-5) por lote de 12 fuentes; sin control de temperature/seed"}
    (A / "generated_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
