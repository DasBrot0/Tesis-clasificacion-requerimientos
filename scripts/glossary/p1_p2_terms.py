#!/usr/bin/env python3
"""P1 y P2 del plan de traducción con glosario (ver notas_sesion/PLAN_TRADUCCION_GLOSARIO.md).

P1. Extrae candidatos a término de los textos únicos en inglés (source_en.csv) con spaCy:
    - sintagmas nominales: (ADJ|NOUN|PROPN)* (NOUN|PROPN), de 1 a 4 tokens, incluidas las
      subsecuencias anidadas (las necesita el C-value);
    - NOUN ADP NOUN (p. ej. "denial of service");
    - siglas: tokens en mayúsculas de 2 o más caracteres (un plural "MACs" cuenta como "MAC").
P2. Calcula frecuencia, dispersión, C-value (multipalabra) y log-likelihood G2 frente al inglés
    general (frecuencias de wordfreq). Filtra por frecuencia >= 2 y G2 > 0 (las siglas pasan
    siempre).

Salidas en datasets/interim/glossary/:
    candidates_raw.csv     todos los candidatos con sus ocurrencias
    candidates_ranked.csv  candidatos que pasan el filtro, con métricas y ranking
    term_occurrences.csv   (término, Source_ID) para P3/P4

Uso:  python scripts/glossary/p1_p2_terms.py [--model en_core_web_trf]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy
from wordfreq import word_frequency

ROOT = Path(__file__).resolve().parents[2]
GLOSSARY_DIR = ROOT / "datasets" / "interim" / "glossary"
MIN_FREQ = 2          # [Decisión propia] ver plan, P2
MAX_LEN = 4
REF_CORPUS_TOKENS = 1e9   # tamaño nominal del corpus de referencia para G2 (wordfreq da proporciones)
WORDFREQ_FLOOR = 1e-9     # frecuencia mínima de wordfreq para palabras ausentes
NOMINAL = {"NOUN", "PROPN"}
MODIFIER = {"ADJ", "NOUN", "PROPN"}
ACRONYM = re.compile(r"^[A-Z][A-Z0-9]*[A-Z0-9](s)?$")
LEMMA_FIXES = {"datum": "data"}  # en este dominio "data" es incontable


def _tok_key(tok) -> str:
    """Forma normalizada del token: siglas en mayúsculas sin plural; resto, lema en minúsculas."""
    m = ACRONYM.match(tok.text)
    if m and len(tok.text) >= 2 and not tok.text.isdigit():
        return tok.text[:-1] if m.group(1) and len(tok.text) > 2 else tok.text
    if tok.pos_ in NOMINAL:
        lemma = tok.lemma_.lower()
        return LEMMA_FIXES.get(lemma, lemma)
    return tok.text.lower()


def _is_acronym(tok) -> bool:
    return bool(ACRONYM.match(tok.text)) and len(tok.text) >= 2 and any(c.isalpha() for c in tok.text)


def extract(doc) -> list[tuple[str, str, int]]:
    """Devuelve (clave, patrón, n_tokens) por ocurrencia en el documento."""
    out: list[tuple[str, str, int]] = []
    toks = [t for t in doc]
    # 1) secuencias maximales de modificadores nominales y todas sus subsecuencias válidas
    i = 0
    while i < len(toks):
        if toks[i].pos_ not in MODIFIER or toks[i].is_punct:
            i += 1
            continue
        j = i
        while j < len(toks) and toks[j].pos_ in MODIFIER and not toks[j].is_punct:
            j += 1
        run = toks[i:j]
        for s in range(len(run)):
            for e in range(s + 1, min(len(run), s + MAX_LEN) + 1):
                span = run[s:e]
                if span[-1].pos_ not in NOMINAL:
                    continue
                if len(span) == 1 and not _is_acronym(span[0]) and (
                        not span[0].is_alpha or len(span[0].text) < 2):
                    continue
                out.append((" ".join(_tok_key(t) for t in span), "NP", len(span)))
        i = j
    # 2) NOUN ADP NOUN
    for k in range(1, len(toks) - 1):
        a, p, b = toks[k - 1], toks[k], toks[k + 1]
        if a.pos_ in NOMINAL and p.pos_ == "ADP" and b.pos_ in NOMINAL:
            out.append((f"{_tok_key(a)} {p.text.lower()} {_tok_key(b)}", "N_ADP_N", 3))
    # 3) siglas (ya incluidas como NP de 1 token si son PROPN/NOUN; se marcan aparte)
    for t in toks:
        if _is_acronym(t):
            out.append((_tok_key(t), "ACRONYM", 1))
    return out


def g2(freq: int, n_corpus: int, ref_prop: float) -> float:
    """Log-likelihood (Rayson y Garside) frente a un corpus de referencia de tamaño nominal."""
    a, b = freq, max(ref_prop, WORDFREQ_FLOOR) * REF_CORPUS_TOKENS
    c, d = n_corpus, REF_CORPUS_TOKENS
    e1, e2 = c * (a + b) / (c + d), d * (a + b) / (c + d)
    val = 2 * ((a * math.log(a / e1) if a else 0) + (b * math.log(b / e2) if b else 0))
    return val if a / c > b / d else -val  # negativo = menos frecuente que en inglés general


def c_value(freqs: dict[str, int], lengths: dict[str, int]) -> dict[str, float]:
    """C-value (Frantzi et al.) para candidatos multipalabra; anidamiento por subcadena de tokens."""
    multi = [t for t in freqs if lengths[t] > 1]
    containers: dict[str, list[str]] = defaultdict(list)
    for longer in multi:
        toks = longer.split()
        for s in range(len(toks)):
            for e in range(s + 1, len(toks) + 1):
                if e - s < len(toks):
                    sub = " ".join(toks[s:e])
                    if sub in freqs and lengths[sub] > 1:
                        containers[sub].append(longer)
    out = {}
    for t in multi:
        base = math.log2(lengths[t])
        nest = set(containers.get(t, []))
        if nest:
            out[t] = base * (freqs[t] - sum(freqs[b] for b in nest) / len(nest))
        else:
            out[t] = base * freqs[t]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="en_core_web_trf")
    args = parser.parse_args()

    src = pd.read_csv(GLOSSARY_DIR / "source_en.csv", encoding="utf-8-sig")
    uniq = src[src["Is_first_occurrence"]].reset_index(drop=True)
    nlp = spacy.load(args.model, disable=["ner"])
    print(f"spaCy {spacy.__version__}, modelo {args.model} {nlp.meta['version']}; {len(uniq)} textos únicos")

    freq: Counter = Counter()
    lengths: dict[str, int] = {}
    patterns: dict[str, set] = defaultdict(set)
    surface: dict[str, Counter] = defaultdict(Counter)
    occ_rows = []
    n_tokens = 0
    for doc, (_, row) in zip(nlp.pipe(uniq["Requirement_en"].tolist(), batch_size=32), uniq.iterrows()):
        n_tokens += sum(1 for t in doc if not t.is_punct and not t.is_space)
        seen = set()
        for key, pattern, n in extract(doc):
            if pattern == "ACRONYM":
                patterns[key].add(pattern)
                continue
            freq[key] += 1
            lengths[key] = n
            patterns[key].add(pattern)
            if (key, row["Source_ID"]) not in seen:
                seen.add((key, row["Source_ID"]))
                occ_rows.append({"term": key, "Source_ID": row["Source_ID"],
                                 "Dataset": row["Dataset"], "Spec": row["Spec"]})
        for t in doc:
            surface[_tok_key(t)][t.text] += 1

    occ = pd.DataFrame(occ_rows)
    occ.to_csv(GLOSSARY_DIR / "term_occurrences.csv", index=False, encoding="utf-8-sig")
    by_term = occ.groupby("term").agg(doc_freq=("Source_ID", "nunique"), n_specs=("Spec", "nunique"),
                                      datasets=("Dataset", lambda s: "|".join(sorted(set(s)))),
                                      example_ids=("Source_ID", lambda s: "|".join(list(s)[:5])))
    raw = pd.DataFrame({"term": list(freq), "freq": [freq[t] for t in freq],
                        "n_tokens": [lengths[t] for t in freq],
                        "pattern": ["|".join(sorted(patterns[t])) for t in freq]}).set_index("term")
    raw = raw.join(by_term)
    raw["is_acronym"] = raw.index.map(lambda t: "ACRONYM" in patterns[t])
    raw.sort_values("freq", ascending=False).to_csv(GLOSSARY_DIR / "candidates_raw.csv",
                                                    encoding="utf-8-sig")

    kept = raw[(raw["freq"] >= MIN_FREQ) | raw["is_acronym"]].copy()
    cv = c_value(kept["freq"].to_dict(), kept["n_tokens"].to_dict())
    kept["c_value"] = kept.index.map(lambda t: cv.get(t, float("nan")))
    kept["ref_freq_wordfreq"] = kept.index.map(lambda t: word_frequency(t, "en"))
    kept["g2"] = [g2(f, n_tokens, r) for f, r in zip(kept["freq"], kept["ref_freq_wordfreq"])]
    # Sin especificidad de dominio (no es más frecuente que en inglés general): fuera, salvo siglas.
    n_before = len(kept)
    kept = kept[(kept["g2"] > 0) | kept["is_acronym"]].copy()
    print(f"descartados por G2 <= 0: {n_before - len(kept)}")
    # Ranking combinado: media de percentiles. Multipalabra: C-value, G2 y dispersión;
    # una palabra: G2 y dispersión (C-value no está definido para un solo token).
    pct = lambda s: s.rank(pct=True)  # noqa: E731
    kept["pct_g2"] = pct(kept["g2"])
    kept["pct_disp"] = pct(kept["doc_freq"])
    multi = kept["n_tokens"] > 1
    kept.loc[multi, "pct_cvalue"] = pct(kept.loc[multi, "c_value"])
    kept["rank_score"] = kept[["pct_cvalue", "pct_g2", "pct_disp"]].mean(axis=1, skipna=True)
    kept = kept.sort_values("rank_score", ascending=False)
    kept["rank"] = range(1, len(kept) + 1)
    kept.to_csv(GLOSSARY_DIR / "candidates_ranked.csv", encoding="utf-8-sig")

    meta = {"spacy": spacy.__version__, "model": args.model, "model_version": nlp.meta["version"],
            "n_texts": len(uniq), "n_tokens": n_tokens, "min_freq": MIN_FREQ,
            "ref_corpus_tokens": REF_CORPUS_TOKENS, "n_raw": len(raw), "n_kept": len(kept)}
    (GLOSSARY_DIR / "p1_p2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
