#!/usr/bin/env python3
"""E2 (corrida provisional): preanotación de subcategorías con SON sobre el texto EN.

    prepare  -> labels/batches/batch_XX.jsonl (+ items.csv con id, dataset, texto)
    validate -> comprueba labels/runs/run_R/batch_XX.jsonl
    merge    -> son_prelabels.csv (voto mayoritario), son_prelabels_run{1,2,3}.csv, agreement.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "datasets" / "interim" / "glossary"
L = ROOT / "datasets" / "interim" / "labels"
GUIDE = ROOT / "notas_sesion" / "guia_anotacion_subcategorias_v1_provisional.md"
PROMPT = L / "prompt_v1.md"
OLD_PROMISE = ROOT / "old" / "run_2026-10_regex_opusmt" / "datasets" / "training" / "promise.csv"
COLS = ["Confidentiality", "Integrity", "Non_repudiation", "Accountability", "Authenticity", "Resistance"]
BATCH, N_RUNS, MODEL = 29, 3, "claude-sonnet-5-5"
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["id", *COLS, "confidence", "rationale"],
          "properties": {"id": {"type": "string"}, "rationale": {"type": "string"},
                         **{c: {"enum": [0, 1]} for c in COLS},
                         "confidence": {"type": "object", "required": COLS, "additionalProperties": False,
                                        "properties": {c: {"type": "number", "minimum": 0, "maximum": 1} for c in COLS}}}}


def _jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def items() -> pd.DataFrame:
    pr = pd.read_csv(OLD_PROMISE, encoding="utf-8-sig")
    pr = pr[pr.Security == 1].drop_duplicates("Requirement_original")
    rows = [{"dataset": "promise", "text": t} for t in pr.Requirement_original]
    src = pd.read_csv(G / "source_en.csv", encoding="utf-8-sig")
    sr = src[(src.Dataset == "secreq")]
    sec = sr[sr.Label == "sec"].drop_duplicates("Requirement_en")
    rows += [{"dataset": "secreq", "text": t} for t in sec.Requirement_en]
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"e{i:04d}" for i in range(1, len(df) + 1)])
    df["key"] = df.text.map(norm)
    return df


def prepare():
    df = items()
    (L / "batches").mkdir(exist_ok=True)
    df.to_csv(L / "items.csv", index=False, encoding="utf-8-sig")
    n = 0
    for i in range(0, len(df), BATCH):
        n += 1
        with open(L / "batches" / f"batch_{n:02d}.jsonl", "w", encoding="utf-8") as fh:
            for r in df.iloc[i:i + BATCH].itertuples():
                fh.write(json.dumps({"id": r.id, "text": r.text}, ensure_ascii=False) + "\n")
    h = hashlib.sha256(PROMPT.read_bytes() + GUIDE.read_bytes()).hexdigest()[:12]
    m = {"model": MODEL, "prompt": PROMPT.name, "guide": GUIDE.name, "prompt_guide_sha256_12": h,
         "n_items": len(df), "by_dataset": df.dataset.value_counts().to_dict(), "n_batches": n, "n_runs": N_RUNS}
    (L / "batches" / "manifest.json").write_text(json.dumps(m, indent=2))
    print(m)


def validate():
    v, ok = Draft202012Validator(SCHEMA), True
    for b in sorted((L / "batches").glob("batch_*.jsonl")):
        exp = [r["id"] for r in _jl(b)]
        for run in range(1, N_RUNS + 1):
            out = L / "runs" / f"run_{run}" / b.name
            if not out.exists():
                print("FALTA", run, b.name); ok = False; continue
            try:
                rows = _jl(out)
            except Exception as e:
                print("JSON", run, b.name, e); ok = False; continue
            errs = [e.message for r in rows for e in v.iter_errors(r)]
            if [r.get("id") for r in rows] != exp:
                errs.append("ids no coinciden")
            if errs:
                print("INVALIDO", run, b.name, errs[:3]); ok = False
    print("todo válido" if ok else "hay lotes por repetir")
    return ok


def merge():
    df = pd.read_csv(L / "items.csv", encoding="utf-8-sig")
    runs = {}
    for run in range(1, N_RUNS + 1):
        rows = [r for f in sorted((L / "runs" / f"run_{run}").glob("batch_*.jsonl")) for r in _jl(f)]
        runs[run] = {r["id"]: r for r in rows}
        pd.DataFrame([{"id": i, **{c: r[c] for c in COLS}, **{f"conf_{c}": r["confidence"][c] for c in COLS},
                       "rationale": r["rationale"]} for i, r in runs[run].items()]).to_csv(
            L / f"son_prelabels_run{run}.csv", index=False, encoding="utf-8-sig")
    out, agree = [], {c: 0 for c in COLS}
    unanim_all = 0
    for r in df.itertuples():
        v = [runs[k][r.id] for k in runs]
        row = {"id": r.id, "dataset": r.dataset, "text": r.text}
        full = True
        for c in COLS:
            s = sum(x[c] for x in v)
            row[c] = int(s >= 2)
            row[f"{c}_votes"] = s
            row[f"conf_{c}"] = round(sum(x["confidence"][c] for x in v) / len(v), 3)
            u = s in (0, len(v))
            agree[c] += int(u)
            full &= u
        unanim_all += int(full)
        row["rationale"] = v[0]["rationale"]
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(L / "son_prelabels.csv", index=False, encoding="utf-8-sig")
    # kappa por pares entre corridas
    from itertools import combinations
    kap = {}
    for c in COLS:
        ks = []
        for a, b in combinations(runs, 2):
            x = [runs[a][i][c] for i in df.id]; y = [runs[b][i][c] for i in df.id]
            po = sum(p == q for p, q in zip(x, y)) / len(x)
            pa, pb = sum(x) / len(x), sum(y) / len(y)
            pe = pa * pb + (1 - pa) * (1 - pb)
            ks.append((po - pe) / (1 - pe) if pe < 1 else 1.0)
        kap[c] = round(sum(ks) / len(ks), 3)
    rep = {"n": len(df), "unanimity_per_label": {c: round(agree[c] / len(df), 4) for c in COLS},
           "unanimity_all_labels": round(unanim_all / len(df), 4), "mean_pairwise_kappa": kap,
           "positives": {d: {c: int(g[c].sum()) for c in COLS} for d, g in res.groupby("dataset")},
           "none_applies": {d: int((g[COLS].sum(axis=1) == 0).sum()) for d, g in res.groupby("dataset")}}
    (L / "agreement_between_runs.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "validate": validate, "merge": merge}[sys.argv[1]]()
