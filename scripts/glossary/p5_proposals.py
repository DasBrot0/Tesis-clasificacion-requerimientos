#!/usr/bin/env python3
"""P5 (corrida provisional): lotes para que SON proponga equivalentes ES, validación y glosario.

    prepare  -> p5/batches/batch_XX.jsonl (term + hasta 2 contextos EN) y manifest.json
    validate -> comprueba p5/runs/batch_XX.jsonl contra el schema
    merge    -> p5/term_proposals.jsonl
    build    -> glossary_en_es_provisional.tsv (primera propuesta, decided_by=SON_provisional)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "datasets" / "interim" / "glossary"
P5 = G / "p5"
PROMPT = P5 / "prompt_v1.md"
BATCH = 29
MODEL = "claude-sonnet-5-5"
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["cand_id", "term", "proposals"],
          "properties": {"cand_id": {"type": "string"}, "term": {"type": "string"},
                         "proposals": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                             "type": "object", "additionalProperties": False,
                             "required": ["es", "type", "likely_source", "note"],
                             "properties": {"es": {"type": "string", "minLength": 1},
                                            "type": {"enum": ["traducción", "préstamo", "sigla_sin_traducir"]},
                                            "likely_source": {"type": "string"},
                                            "note": {"type": "string"}}}}}}


def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _contexts(ids, texts, specs):
    chosen, seen = [], set()
    for s in ids:
        if specs[s] not in seen:
            chosen.append(s)
            seen.add(specs[s])
        if len(chosen) == 2:
            break
    for s in ids:
        if len(chosen) == 2:
            break
        if s not in chosen:
            chosen.append(s)
    return [texts[s][:300] for s in chosen]


def prepare():
    sl = pd.read_csv(G / "p5_shortlist.csv", encoding="utf-8-sig")
    occ = pd.read_csv(G / "term_occurrences.csv", encoding="utf-8-sig")
    src = pd.read_csv(G / "source_en.csv", encoding="utf-8-sig")
    texts = dict(zip(src.Source_ID, src.Requirement_en))
    specs = dict(zip(src.Source_ID, src.Spec))
    ids = occ.groupby("term")["Source_ID"].apply(list).to_dict()
    rows = [{"cand_id": r.cand_id, "term": r.term, "freq": int(r.freq), "category": r.category,
             "contexts": _contexts(ids.get(r.term, []), texts, specs)} for r in sl.itertuples()]
    (P5 / "batches").mkdir(exist_ok=True)
    n = 0
    for i in range(0, len(rows), BATCH):
        n += 1
        with open(P5 / "batches" / f"batch_{n:02d}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows[i:i + BATCH]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    m = {"model": MODEL, "prompt": PROMPT.name,
         "prompt_sha256_12": hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:12],
         "n_batches": n, "n_terms": len(rows)}
    (P5 / "batches" / "manifest.json").write_text(json.dumps(m, indent=2))
    print(m)


def validate():
    v = Draft202012Validator(SCHEMA)
    ok = True
    for b in sorted((P5 / "batches").glob("batch_*.jsonl")):
        exp = [r["cand_id"] for r in _jl(b)]
        out = P5 / "runs" / b.name
        if not out.exists():
            print("FALTA", b.name)
            ok = False
            continue
        try:
            rows = _jl(out)
        except Exception as e:
            print("JSON", b.name, e)
            ok = False
            continue
        errs = [e.message for r in rows for e in v.iter_errors(r)]
        if [r.get("cand_id") for r in rows] != exp:
            errs.append("ids no coinciden")
        if errs:
            print("INVALIDO", b.name, errs[:3])
            ok = False
    print("todo válido" if ok else "hay lotes por repetir")
    return ok


def merge():
    with open(P5 / "term_proposals.jsonl", "w", encoding="utf-8") as fh:
        for b in sorted((P5 / "runs").glob("batch_*.jsonl")):
            for r in _jl(b):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# Entradas defectuosas de la extracción automática (fragmentos truncados, palabras comunes cuya
# primera propuesta altera el sentido). Se descartan SIN criterio terminológico experto: es una
# limpieza mínima para que el glosario provisional no dañe la traducción. Registradas en build_log.
DROP = {"manage", "secure", "ES", "client re", "user re", "theIntrusionD", "Zimbabwe Anti", "contract",
        "template", "bit encryption key", "bit encryption", "current card I"}
PREPS = {"de", "del", "en", "para", "con", "a", "por", "al", "sobre", "sin", "entre"}


def _plural_word(w):
    lw = w.lower()
    if not w.isalpha() or w.isupper() or lw.endswith("s") and not lw.endswith(("és", "ás", "ís", "ós", "ús")):
        return w
    if lw.endswith("ón"):
        return w[:-2] + "ones"
    if lw.endswith(("é", "á", "ó", "a", "e", "o", "í", "ú")):
        return w + ("es" if lw.endswith(("í", "ú")) else "s")
    if lw.endswith("z"):
        return w[:-1] + "ces"
    if lw.endswith(("és", "ás", "ís", "ós", "ús")):
        return w[:-2] + {"é": "e", "á": "a", "í": "i", "ó": "o", "ú": "u"}[w[-2]] + "ses"
    return w + "es"


def pluralize_es(phrase):
    out, before_prep = [], True
    for w in phrase.split():
        if w.lower() in PREPS:
            before_prep = False
        out.append(_plural_word(w) if before_prep and w.lower() not in {"no", "muy", "más"} else w)
    return " ".join(out)


def build():
    src = pd.read_csv(G / "source_en.csv", encoding="utf-8-sig")
    texts = src.drop_duplicates("Requirement_en")["Requirement_en"].tolist()
    sl = pd.read_csv(G / "p5_shortlist.csv", encoding="utf-8-sig")
    props = {r["cand_id"]: r for r in _jl(P5 / "term_proposals.jsonl")}
    rows, seen, log = [], {}, []
    for r in sl.itertuples():
        lemma = r.term
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(lemma).replace(r"\ ", r"[\s-]+")
                         + r"(?:s|es)?(?![A-Za-z0-9])", re.I)
        cnt = Counter()
        for t in texts:
            for m in pat.finditer(t):
                cnt[re.sub(r"\s+", " ", m.group(0))] += 1
        surf = cnt.most_common(1)[0][0] if cnt else lemma
        # preferir la forma singular si es al menos la mitad de frecuente que la más común
        sing = [k for k in cnt if k.lower().replace("-", " ") == lemma.lower()]
        if sing and max(cnt[k] for k in sing) * 2 >= cnt[surf]:
            surf = max(sing, key=lambda k: cnt[k])
        p = props[r.cand_id]["proposals"][0]
        es = " ".join(p["es"].split())
        if r.category == "NE" or p["type"] == "sigla_sin_traducir":
            es = surf  # mismo texto de origen: no se traduce
        key = surf.strip()
        if not key or not es:
            continue
        if key in DROP or "(" in key:
            log.append(f"descartada (entrada defectuosa): {key!r} -> {es!r}")
            continue
        is_plural_surface = key.lower().replace("-", " ") != lemma.lower() and key.lower().endswith("s")
        if is_plural_surface and r.category != "NE" and p["type"] == "traducción" and es.lower() != key.lower():
            es = pluralize_es(es)  # la forma del corpus es plural: el equivalente también
        if key in seen:
            log.append(f"duplicado omitido: {key!r} (lema {r.term}) ya definido por {seen[key]!r}")
            continue
        seen[key] = r.term
        note = f"lema={lemma}; {p['note']}"[:300].replace("\t", " ").replace("\n", " ")
        rows.append({"en": key, "es": es, "type": p["type"], "source": "SON_provisional",
                     "source_ref": "", "decided_by": "SON_provisional", "date": "2026-10-18",
                     "notes": note})
    df = pd.DataFrame(rows)
    df.to_csv(G / "glossary_en_es_provisional.tsv", sep="\t", index=False, encoding="utf-8")
    (P5 / "build_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(len(df), "entradas;", len(log), "duplicados omitidos")


if __name__ == "__main__":
    {"prepare": prepare, "validate": validate, "merge": merge, "build": build}[sys.argv[1]]()
