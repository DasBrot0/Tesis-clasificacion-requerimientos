#!/usr/bin/env python3
"""P8 (corrida provisional): exactitud terminológica + CometKiwi + LaBSE sobre la traducción DeepL.

    python scripts/glossary/p8_qc.py terms     -> translations/_qc_terms.csv (term_acc, term_missing)
    (luego) cometkiwi/LaBSE en el venv aislado -> translations/_qc_neural.csv
    python scripts/glossary/p8_qc.py merge     -> qc_signals_provisional.csv + qc_summary_provisional.md
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "inputs" / "interim" / "glossary"
T = ROOT / "inputs" / "interim" / "translations"
VENV = "C:/Users/Franco/.venvs/cometkiwi/Scripts/python.exe"


def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def load():
    rows = [json.loads(l) for l in open(T / "deepl_provisional.jsonl", encoding="utf-8")]
    return pd.DataFrame(rows)


def terms():
    import spacy
    nlp = spacy.load("es_core_news_lg", disable=["parser", "ner"])
    gl = pd.read_csv(G / "glossary_en_es_provisional.tsv", sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
    pats = [(r.en, r.es, re.compile(r"(?<![A-Za-z0-9])" + re.escape(r.en).replace(r"\ ", r"[\s-]+") + r"(?:s|es)?(?![A-Za-z0-9])", re.I))
            for r in gl.itertuples()]
    df = load()
    lemma_cache = {}

    def lem(s):
        if s not in lemma_cache:
            lemma_cache[s] = " ".join(strip_acc(t.lemma_) for t in nlp(s) if not t.is_punct)
        return lemma_cache[s]

    out = []
    es_docs = list(nlp.pipe(df["es"].tolist(), batch_size=64))
    for r, doc in zip(df.itertuples(), es_docs):
        es_norm = strip_acc(r.es)
        es_lem = " " + " ".join(strip_acc(t.lemma_) for t in doc if not t.is_punct) + " "
        present, found_exact, found_lemma, missing = 0, 0, 0, []
        for en, es, p in pats:
            if p.search(r.en):
                present += 1
                ex = strip_acc(es) in es_norm
                lm = (" " + lem(es) + " ") in es_lem
                found_exact += ex
                found_lemma += (ex or lm)
                if not (ex or lm):
                    missing.append(f"{en}->{es}")
        out.append({"en": r.en, "n_terms": present,
                    "term_acc_exact": found_exact / present if present else 1.0,
                    "term_acc": found_lemma / present if present else 1.0,
                    "term_missing": "; ".join(missing)})
    pd.DataFrame(out).to_csv(T / "_qc_terms.csv", index=False, encoding="utf-8-sig")
    df[["en", "es"]].to_csv(T / "_qc_neural_in.csv", index=False, encoding="utf-8-sig")
    print("términos listos")


def neural():
    subprocess.run([VENV, str(Path(__file__).with_name("p8_neural_signals.py")),
                    str(T / "_qc_neural_in.csv"), str(T / "_qc_neural.csv")], check=True)


def merge():
    df = load()
    df = df.merge(pd.read_csv(T / "_qc_terms.csv", encoding="utf-8-sig"), on="en")
    df = df.merge(pd.read_csv(T / "_qc_neural.csv", encoding="utf-8-sig")[["en", "cometkiwi", "labse"]], on="en")
    df["Source_ID"] = df["Source_IDs"].map(lambda x: x[0])
    df["Source_IDs"] = df["Source_IDs"].map(lambda x: "|".join(x))
    cols = ["Source_ID", "Source_IDs", "Dup_group", "term_acc", "term_acc_exact", "n_terms", "term_missing",
            "cometkiwi", "labse", "en", "es"]
    df[cols].to_csv(T / "qc_signals_provisional.csv", index=False, encoding="utf-8-sig")
    low_labse = (df.labse < 0.80).sum()
    lines = [
        "# Control de calidad provisional (DeepL + glosario)", "",
        f"- Textos únicos: {len(df)}",
        f"- CometKiwi: media {df.cometkiwi.mean():.4f}, mediana {df.cometkiwi.median():.4f}, "
        f"p5 {df.cometkiwi.quantile(.05):.4f}, mín {df.cometkiwi.min():.4f}, máx {df.cometkiwi.max():.4f}",
        f"- Exactitud terminológica por lema == 1: {(df.term_acc == 1).mean() * 100:.1f}% "
        f"(exacta: {(df.term_acc_exact == 1).mean() * 100:.1f}%); con al menos un término: {(df.n_terms > 0).sum()}",
        f"- Exactitud terminológica media (solo requisitos con términos): {df[df.n_terms > 0].term_acc.mean():.4f}",
        f"- LaBSE: media {df.labse.mean():.4f}; < 0,80: {low_labse} ({low_labse / len(df) * 100:.1f}%)", "",
        "## Peores 20 por CometKiwi", ""]
    for r in df.nsmallest(20, "cometkiwi").itertuples():
        lines.append(f"- {r.cometkiwi:.3f} | acc {r.term_acc:.2f} | labse {r.labse:.2f} | EN: {r.en[:150]} | ES: {r.es[:150]}")
    lines += ["", "## Peores 20 por exactitud terminológica (más términos faltantes)", ""]
    for r in df[df.n_terms > 0].assign(nm=lambda d: d.n_terms * (1 - d.term_acc)).nlargest(20, "nm").itertuples():
        lines.append(f"- acc {r.term_acc:.2f} | faltan: {r.term_missing} | EN: {r.en[:120]}")
    (T / "qc_summary_provisional.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    {"terms": terms, "neural": neural, "merge": merge}[sys.argv[1]]()
