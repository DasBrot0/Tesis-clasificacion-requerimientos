#!/usr/bin/env python3
"""P4 del plan de traducción con glosario: cómo tradujeron los candidatos las traducciones previas.

Para cada requisito (texto único en inglés) se alinean palabras EN-ES de la traducción de base
(opus-mt para DOSSPRE; old/SecReq_es.csv para SecReq) con embeddings contextuales de LaBSE
(capa intermedia) e intersección de argmax en ambas direcciones (Wang et al., 2022). La
traducción de cada ocurrencia de un candidato es el tramo español cubierto por las palabras
alineadas con él, normalizado a lemas en minúsculas.

Salidas en datasets/interim/glossary/:
    term_baseline_renderings.csv     una fila por (término, requisito) con su traducción de base
    term_inconsistency_baseline.csv  por término: ocurrencias, variantes, consistencia, flags

Uso:  python scripts/glossary/p4_baseline_consistency.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "glossary"))
from p1_p2_terms import _tok_key  # noqa: E402

G = ROOT / "datasets" / "interim" / "glossary"
ALIGNER = "sentence-transformers/LaBSE"
LAYER = 6  # [Decisión propia] capa intermedia; Wang et al. (2022) reportan mejores alineamientos en capas medias


def word_embeddings(words: list[str], tok, model, device) -> np.ndarray:
    enc = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True,
              max_length=256).to(device)
    with torch.no_grad():
        hidden = model(**enc, output_hidden_states=True).hidden_states[LAYER][0]
    ids = enc.word_ids()
    vecs = np.zeros((len(words), hidden.shape[-1]), dtype=np.float32)
    counts = np.zeros(len(words))
    for pos, wid in enumerate(ids):
        if wid is not None:
            vecs[wid] += hidden[pos].cpu().numpy()
            counts[wid] += 1
    vecs[counts > 0] /= counts[counts > 0, None]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def align(en_vec: np.ndarray, es_vec: np.ndarray) -> set[tuple[int, int]]:
    sim = en_vec @ es_vec.T
    fwd = {(i, int(sim[i].argmax())) for i in range(sim.shape[0])}
    bwd = {(int(sim[:, j].argmax()), j) for j in range(sim.shape[1])}
    return fwd & bwd


def find_spans(keys: list[str], term: str) -> list[tuple[int, int]]:
    parts = term.split()
    n = len(parts)
    return [(i, i + n) for i in range(len(keys) - n + 1) if keys[i:i + n] == parts]


def main() -> None:
    src = pd.read_csv(G / "source_en.csv", encoding="utf-8-sig")
    uniq = src[src["Is_first_occurrence"]].reset_index(drop=True)
    terms = set(pd.read_csv(G / "candidates_ranked.csv", encoding="utf-8-sig")["term"])
    occ = pd.read_csv(G / "term_occurrences.csv", encoding="utf-8-sig")
    occ = occ[occ["term"].isin(terms)]
    terms_by_id = occ.groupby("Source_ID")["term"].apply(list).to_dict()

    nlp_en = spacy.load("en_core_web_trf", disable=["ner"])
    nlp_es = spacy.load("es_core_news_lg", disable=["ner", "parser"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ALIGNER)
    model = AutoModel.from_pretrained(ALIGNER).to(device).eval()

    rows = []
    en_docs = nlp_en.pipe(uniq["Requirement_en"].tolist(), batch_size=32)
    es_docs = nlp_es.pipe(uniq["ES_baseline"].fillna("").tolist(), batch_size=64)
    for k, (row, en_doc, es_doc) in enumerate(zip(uniq.itertuples(index=False), en_docs, es_docs)):
        wanted = terms_by_id.get(row.Source_ID)
        if not wanted:
            continue
        en_toks = [t for t in en_doc if not t.is_space]
        es_toks = [t for t in es_doc if not t.is_space]
        if not es_toks:
            continue
        keys = [_tok_key(t) for t in en_toks]
        links = align(word_embeddings([t.text for t in en_toks], tok, model, device),
                      word_embeddings([t.text for t in es_toks], tok, model, device))
        for term in set(wanted):
            for s, e in find_spans(keys, term):
                tgt = sorted(j for i, j in links if s <= i < e)
                if tgt:
                    span = es_toks[tgt[0]:tgt[-1] + 1]
                    surface = " ".join(t.text for t in span)
                    # Solo los sustantivos se lematizan (singular); el resto queda en superficie
                    # para no alterar género de adjetivos ni separar contracciones ("del").
                    norm = " ".join((t.lemma_ if t.pos_ in ("NOUN", "PROPN") else t.text).lower()
                                    for t in span if not t.is_punct)
                else:
                    surface, norm = "", "<sin alinear>"
                rows.append({"term": term, "Source_ID": row.Source_ID, "Dataset": row.Dataset,
                             "baseline": row.ES_baseline_source,
                             "en_span": " ".join(t.text for t in en_toks[s:e]),
                             "es_surface": surface, "es_norm": norm,
                             "aligned_ratio": round(len({i for i, _ in links if s <= i < e}) / (e - s), 2)})
        if k % 200 == 0:
            print(f"  {k}/{len(uniq)}", flush=True)

    rend = pd.DataFrame(rows)
    rend.to_csv(G / "term_baseline_renderings.csv", index=False, encoding="utf-8-sig")

    summary = []
    for (term, dataset), g in rend.groupby(["term", "Dataset"]):
        valid = g[g["es_norm"] != "<sin alinear>"]
        variants = Counter(valid["es_norm"])
        top, top_n = variants.most_common(1)[0] if variants else ("", 0)
        en_words = [w for w in term.lower().split() if len(w) > 2]
        left_en = sum(1 for s in valid["es_surface"].str.lower()
                      if en_words and all(re.search(rf"\b{re.escape(w)}\b", s) for w in en_words))
        summary.append({
            "term": term, "Dataset": dataset, "n_occurrences": len(g),
            "n_unaligned": int((g["es_norm"] == "<sin alinear>").sum()),
            "n_variants": len(variants), "top_variant": top,
            "consistency": round(top_n / len(valid), 3) if len(valid) else None,
            "variants": json.dumps(dict(variants.most_common(6)), ensure_ascii=False),
            "n_left_in_english": left_en,
        })
    summ = pd.DataFrame(summary)
    summ["flag_inconsistent"] = (summ["n_variants"] >= 2) & (summ["consistency"] < 1)
    summ["flag_left_in_english"] = summ["n_left_in_english"] > 0
    summ.sort_values(["flag_inconsistent", "n_occurrences"], ascending=False).to_csv(
        G / "term_inconsistency_baseline.csv", index=False, encoding="utf-8-sig")
    multi = summ[summ["n_occurrences"] >= 2]
    report = {"aligner": ALIGNER, "layer": LAYER, "n_renderings": len(rend),
              "terms_with_2plus_occ": len(multi),
              "pct_inconsistent_2plus": round(float(multi["flag_inconsistent"].mean()), 3),
              "mean_consistency_2plus": round(float(multi["consistency"].mean()), 3),
              "by_dataset": multi.groupby("Dataset")["consistency"].mean().round(3).to_dict()}
    (G / "p4_meta.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
