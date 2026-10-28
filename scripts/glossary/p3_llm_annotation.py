#!/usr/bin/env python3
"""P3 del plan de traducción con glosario: LLM (subagentes de Claude Code) como segundo anotador.

    prepare   -> p3/batches/batch_XX.jsonl (candidatos con contexto) y p3/batches/manifest.json
    validate  -> comprueba p3/runs/run_R/batch_XX.jsonl contra el schema y la entrada
    merge     -> voto mayoritario de las corridas: candidates_llm.csv (+ acuerdo entre corridas)
    sample    -> human_sample.csv: 200 candidatos estratificados para anotación humana (kappa)

Los subagentes leen p3/prompt_v1.md y un archivo de lote, y escriben la salida en p3/runs/.

Uso:  python scripts/glossary/p3_llm_annotation.py {prepare|validate|merge|sample}
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "inputs" / "interim" / "glossary"
P3 = G / "p3"
PROMPT = P3 / "prompt_v1.md"
BATCH_SIZE = 90
N_RUNS = 3
MODEL = "claude-sonnet-5-5"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cand_id", "term", "category", "is_security_term", "iso25010_subchar",
                 "boundary_ok", "suggested_form", "rationale"],
    "properties": {
        "cand_id": {"type": "string"},
        "term": {"type": "string"},
        "category": {"enum": ["Specific", "Common", "OOD", "NE", "NotTerm"]},
        "is_security_term": {"type": "boolean"},
        "iso25010_subchar": {"enum": ["Confidentiality", "Integrity", "Non_repudiation",
                                      "Accountability", "Authenticity", "Resistance", "none"]},
        "boundary_ok": {"type": "boolean"},
        "suggested_form": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}
VOTE_FIELDS = ["category", "is_security_term", "iso25010_subchar", "boundary_ok"]


def _contexts(term_ids: list[str], texts: dict[str, str], specs: dict[str, str]) -> list[str]:
    chosen, seen_specs = [], set()
    for sid in term_ids:  # prioriza especificaciones distintas
        if specs[sid] not in seen_specs:
            chosen.append(sid)
            seen_specs.add(specs[sid])
        if len(chosen) == 2:
            break
    for sid in term_ids:
        if len(chosen) == 2:
            break
        if sid not in chosen:
            chosen.append(sid)
    return [texts[s][:300] for s in chosen]


def prepare() -> None:
    cands = pd.read_csv(G / "candidates_ranked.csv", encoding="utf-8-sig")
    occ = pd.read_csv(G / "term_occurrences.csv", encoding="utf-8-sig")
    src = pd.read_csv(G / "source_en.csv", encoding="utf-8-sig")
    texts = dict(zip(src["Source_ID"], src["Requirement_en"]))
    specs = dict(zip(src["Source_ID"], src["Spec"]))
    ids_by_term = occ.groupby("term")["Source_ID"].apply(list).to_dict()
    rows = []
    for i, r in enumerate(cands.itertuples(index=False), start=1):
        rows.append({"cand_id": f"c{i:04d}", "term": r.term, "freq": int(r.freq),
                     "contexts": _contexts(ids_by_term.get(r.term, []), texts, specs)})
    (P3 / "batches").mkdir(parents=True, exist_ok=True)
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    for b, chunk in enumerate(batches, start=1):
        with open(P3 / "batches" / f"batch_{b:02d}.jsonl", "w", encoding="utf-8") as fh:
            for row in chunk:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    prompt_hash = hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:12]
    manifest = {"model": MODEL, "prompt": PROMPT.name, "prompt_sha256_12": prompt_hash,
                "batch_size": BATCH_SIZE, "n_batches": len(batches), "n_candidates": len(rows),
                "n_runs": N_RUNS}
    (P3 / "batches" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rows).drop(columns="contexts").to_csv(P3 / "cand_ids.csv", index=False,
                                                       encoding="utf-8-sig")
    print(json.dumps(manifest, indent=2))


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate() -> bool:
    validator = Draft202012Validator(SCHEMA)
    ok = True
    for batch in sorted((P3 / "batches").glob("batch_*.jsonl")):
        expected = [r["cand_id"] for r in _read_jsonl(batch)]
        for run in range(1, N_RUNS + 1):
            out = P3 / "runs" / f"run_{run}" / batch.name
            if not out.exists():
                print(f"FALTA   run_{run}/{batch.name}")
                ok = False
                continue
            try:
                rows = _read_jsonl(out)
            except json.JSONDecodeError as exc:
                print(f"JSON    run_{run}/{batch.name}: {exc}")
                ok = False
                continue
            errs = [f"{r.get('cand_id')}: {e.message}" for r in rows for e in validator.iter_errors(r)]
            got = [r.get("cand_id") for r in rows]
            if got != expected:
                missing = sorted(set(expected) - set(got))
                extra = sorted(set(got) - set(expected))
                errs.append(f"ids no coinciden (faltan {missing[:5]}…, sobran {extra[:5]}…, "
                            f"{len(got)} vs {len(expected)})")
            if errs:
                ok = False
                print(f"INVÁLIDO run_{run}/{batch.name}: {errs[:3]}")
    print("todo válido" if ok else "hay lotes por repetir")
    return ok


def merge() -> None:
    ids = pd.read_csv(P3 / "cand_ids.csv", encoding="utf-8-sig")
    runs = {}
    for run in range(1, N_RUNS + 1):
        rows = [r for f in sorted((P3 / "runs" / f"run_{run}").glob("batch_*.jsonl"))
                for r in _read_jsonl(f)]
        runs[run] = {r["cand_id"]: r for r in rows}
    out = []
    agreement = Counter()
    for cid in ids["cand_id"]:
        votes = [runs[r][cid] for r in runs if cid in runs[r]]
        row = {"cand_id": cid}
        for field in VOTE_FIELDS:
            values = [v[field] for v in votes]
            value, count = Counter(values).most_common(1)[0]
            row[field] = value
            row[f"{field}_votes"] = count
            agreement[field] += int(count == len(values))
        row["suggested_form"] = next((v["suggested_form"] for v in votes if v["suggested_form"]), None)
        row["rationale"] = votes[0]["rationale"]
        out.append(row)
    merged = ids.merge(pd.DataFrame(out), on="cand_id")
    merged.to_csv(G / "candidates_llm.csv", index=False, encoding="utf-8-sig")
    n = len(merged)
    report = {f: round(agreement[f] / n, 4) for f in VOTE_FIELDS}
    report["n_candidates"] = n
    report["category_counts"] = merged["category"].value_counts().to_dict()
    report["n_security_terms"] = int(merged["is_security_term"].sum())
    (P3 / "agreement_between_runs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def sample() -> None:
    merged = pd.read_csv(G / "candidates_llm.csv", encoding="utf-8-sig")
    parts = [g.sample(min(len(g), max(10, round(200 * len(g) / len(merged)))), random_state=42)
             for _, g in merged.groupby("category")]
    s = pd.concat(parts).sample(frac=1, random_state=42).head(200)
    s = s[["cand_id", "term", "freq"]].assign(category="", is_security_term="", iso25010_subchar="")
    s.to_csv(G / "human_sample.csv", index=False, encoding="utf-8-sig")
    print(f"{len(s)} candidatos -> {G / 'human_sample.csv'} (a completar por el autor, sin ver al LLM)")


if __name__ == "__main__":
    {"prepare": prepare, "validate": validate, "merge": merge, "sample": sample}[sys.argv[1]]()
