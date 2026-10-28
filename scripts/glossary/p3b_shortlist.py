#!/usr/bin/env python3
"""Cruza P2 (ranking) y P3 (etiquetas del LLM) y genera la lista de términos que pasa a P5 (fijar equivalentes).

Entra a P5 un candidato si cumple alguna de estas condiciones:
  A. category == "Specific"
  B. is_security_term == True y category != "NotTerm"
  C. category == "NE" (se fija explícitamente que no se traduce)

Las traducciones previas no intervienen (el antiguo criterio D quedó en old/glossary_p4/).

Salida: inputs/interim/glossary/p5_shortlist.csv

Uso:  python scripts/glossary/p3b_shortlist.py
"""
from pathlib import Path

import pandas as pd

G = Path(__file__).resolve().parents[2] / "inputs" / "interim" / "glossary"

ranked = pd.read_csv(G / "candidates_ranked.csv", encoding="utf-8-sig")
llm = pd.read_csv(G / "candidates_llm.csv", encoding="utf-8-sig")
df = llm.merge(ranked[["term", "doc_freq", "n_specs", "datasets", "c_value", "g2", "rank"]], on="term")
is_term = df["category"] != "NotTerm"
df["reason"] = ""
for code, mask in {
    "A_specific": df["category"].eq("Specific"),
    "B_security": df["is_security_term"] & is_term,
    "C_named_entity": df["category"].eq("NE"),
}.items():
    df.loc[mask, "reason"] += code + "|"
short = df[df["reason"] != ""].copy()
short["reason"] = short["reason"].str.rstrip("|")
short = short.sort_values(["is_security_term", "rank"], ascending=[False, True])
short.to_csv(G / "p5_shortlist.csv", index=False, encoding="utf-8-sig")
print(f"{len(short)} términos a P5 de {len(df)} candidatos")
for code in ["A_specific", "B_security", "C_named_entity"]:
    print(f"  {code}: {short['reason'].str.contains(code).sum()}")
print(f"  seguridad: {int(short['is_security_term'].sum())}")
