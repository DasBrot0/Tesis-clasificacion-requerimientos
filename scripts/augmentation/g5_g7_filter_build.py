#!/usr/bin/env python3
"""G4 (descarte), G5, G6 y G7: filtra los sintéticos y escribe synthetic_<dataset>.csv.

    python scripts/augmentation/g5_g7_filter_build.py

Orden de los filtros (el primer motivo que aplica queda registrado):
  G3  sin verbo modal, o reutiliza 4 palabras seguidas de la fuente
  G4  traducción marcada por P8: CometKiwi o LaBSE por debajo del percentil 5 del corpus real,
      o exactitud terminológica < 0,5                                       [Decisión propia]
  G5  fuga: coseno LaBSE >= 0,90 con algún requisito real del dataset (inglés o español)
  G5  casi copia: coseno LaBSE con su fuente >= 0,85
  G6  el voto mayoritario de las 3 corridas de SON-V no coincide exactamente con las
      subcategorías pretendidas, o alguna etiqueta queda "unclear" en 2 o más corridas

Versión 2 (26-10-2026): la casi copia pasó del tercer cuartil del dataset a un umbral fijo y se
quitó el tope de k_accept por fuente. Con LaBSE el tercer cuartil cae en 0,60-0,64, donde los
requisitos ya no son copias sino escenarios distintos del mismo tema (Limaylla-Lunarejo et al.,
2026, obtienen 0,95 con word2vec); solo 5 de 382 superaban 0,75. Se aceptan entonces las k_request
variantes pedidas por fuente (2 o 3) que pasen los filtros.               [Decisión propia]

Corrida provisional: sin revisión de EXP1 (Label_verified_by = SON-V_x3_provisional).
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import AUGMENTATION_DIR, DATASETS, LABEL_COLS, SUBCATEGORY_COLS, TEXT_COL  # noqa: E402

A = AUGMENTATION_DIR
REAL_QC = ROOT / "inputs" / "interim" / "translations" / "qc_signals_provisional.csv"
LEAK_THRESHOLD = 0.90
NEAR_COPY_THRESHOLD = 0.85
N_RUNS = 3
SCHEMA = {"type": "object", "required": ["id", *SUBCATEGORY_COLS, "confidence", "unclear", "rationale"],
          "properties": {"id": {"type": "string"}, "rationale": {"type": "string"},
                         **{c: {"enum": [0, 1]} for c in SUBCATEGORY_COLS},
                         "unclear": {"type": "array", "items": {"enum": SUBCATEGORY_COLS}},
                         "confidence": {"type": "object", "required": SUBCATEGORY_COLS}}}


def _jl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def verification() -> tuple[pd.DataFrame, dict]:
    validator = Draft202012Validator(SCHEMA)
    runs = {}
    for run in range(1, N_RUNS + 1):
        rows = {}
        for batch in sorted((A / "ver_batches").glob("batch_*.jsonl")):
            expected = [r["id"] for r in _jl(batch)]
            got = _jl(A / "ver_runs" / f"run_{run}" / batch.name)
            errors = [e.message for r in got for e in validator.iter_errors(r)]
            if [r["id"] for r in got] != expected or errors:
                raise SystemExit(f"run_{run}/{batch.name} inválido: {errors[:3] or 'ids no coinciden'}")
            rows.update({r["id"]: r for r in got})
        runs[run] = rows
    ids = sorted(runs[1])
    out = []
    for gen_id in ids:
        votes = [runs[k][gen_id] for k in runs]
        row = {"gen_id": gen_id}
        for c in SUBCATEGORY_COLS:
            row[f"son_{c}"] = int(sum(v[c] for v in votes) >= 2)
            row[f"son_{c}_votes"] = sum(v[c] for v in votes)
        row["son_unclear"] = "|".join(c for c in SUBCATEGORY_COLS if sum(c in v["unclear"] for v in votes) >= 2)
        row["son_rationale"] = votes[0]["rationale"]
        out.append(row)
    kappa = {}
    for c in SUBCATEGORY_COLS:
        ks = []
        for a, b in combinations(runs, 2):
            x = np.array([runs[a][i][c] for i in ids]); y = np.array([runs[b][i][c] for i in ids])
            po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
            ks.append((po - pe) / (1 - pe) if pe < 1 else 1.0)
        kappa[c] = round(float(np.mean(ks)), 3)
    return pd.DataFrame(out), kappa


def labse(texts: list[str]) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    outs = []
    for i in range(0, len(texts), 64):
        batch = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outs.append(torch.nn.functional.normalize(model(**batch).pooler_output, dim=1).cpu())
    return torch.cat(outs)


def main() -> None:
    gen = pd.read_csv(A / "generated_raw.csv", encoding="utf-8-sig")
    sources = pd.read_csv(A / "sources.csv", encoding="utf-8-sig")
    qc = pd.read_csv(A / "qc_signals.csv", encoding="utf-8-sig")
    gen = gen.merge(qc[["gen_id", "es", "term_acc", "cometkiwi", "labse"]], on="gen_id", how="left")
    if gen["es"].isna().any():
        raise SystemExit("faltan traducciones o señales de calidad")
    ver, kappa = verification()
    gen = gen.merge(ver, on="gen_id", how="left", validate="one_to_one")
    gen = gen.merge(sources[["anchor_id", "k_accept"]], on="anchor_id")

    real_qc = pd.read_csv(REAL_QC, encoding="utf-8-sig")
    ck_min, lb_min = real_qc["cometkiwi"].quantile(0.05), real_qc["labse"].quantile(0.05)

    # Similarities: synthetic vs its source and vs every real requirement of its dataset.
    real = {d: pd.read_csv(p, encoding="utf-8-sig", dtype={"ID": str}) for d, p in DATASETS.items()}
    gen["sim_source"] = np.nan
    gen["sim_max_real"] = np.nan
    gen["nearest_real_id"] = ""
    for dataset, part in gen.groupby("dataset"):
        df = real[dataset]
        e_gen_en, e_gen_es = labse(part["requirement_en"].tolist()), labse(part["es"].tolist())
        e_real_en, e_real_es = labse(df["Requirement_original"].tolist()), labse(df[TEXT_COL].tolist())
        sim = torch.maximum(e_gen_en @ e_real_en.T, e_gen_es @ e_real_es.T).numpy()
        pos = {rid: k for k, rid in enumerate(df["ID"])}
        src_idx = part["anchor_id"].map(pos).to_numpy()
        gen.loc[part.index, "sim_source"] = sim[np.arange(len(part)), src_idx]
        gen.loc[part.index, "sim_max_real"] = sim.max(axis=1)
        gen.loc[part.index, "nearest_real_id"] = df["ID"].to_numpy()[sim.argmax(axis=1)]
    q3 = gen.groupby("dataset")["sim_source"].quantile(0.75)  # solo se reporta (versión 1)

    def reason(r) -> str:
        if not r.has_modal or r.reuses_4gram:
            return "G3_formato_o_reuso"
        if r.cometkiwi < ck_min or r.labse < lb_min or r.term_acc < 0.5:
            return "G4_traduccion"
        if r.sim_max_real >= LEAK_THRESHOLD:
            return "G5_fuga_similitud_real"
        if r.sim_source >= NEAR_COPY_THRESHOLD:
            return "G5_casi_copia_fuente"
        intended = set(r.labels.split("|"))
        voted = {c for c in SUBCATEGORY_COLS if getattr(r, f"son_{c}")}
        if voted != intended:
            return "G6_etiqueta_no_coincide"
        if r.son_unclear:
            return "G6_unclear"
        return ""

    gen["reject_reason"] = gen.apply(reason, axis=1)
    gen.to_csv(A / "filter_log.csv", index=False, encoding="utf-8-sig")

    summary = {"filter_version": 2,
               "thresholds": {"cometkiwi_p5_real": round(ck_min, 4), "labse_p5_real": round(lb_min, 4),
                              "leak_cosine": LEAK_THRESHOLD, "near_copy_cosine": NEAR_COPY_THRESHOLD,
                              "source_sim_q3_reported_only": q3.round(4).to_dict()},
               "son_v_mean_pairwise_kappa": kappa, "by_dataset": {}}
    for dataset, part in gen.groupby("dataset"):
        acc = part[part["reject_reason"].eq("")].reset_index(drop=True)
        df = real[dataset].set_index("ID")
        out = pd.DataFrame({
            "ID": [f"syn_{dataset}_{i:04d}" for i in range(1, len(acc) + 1)],
            "Dataset": dataset, "Anchor_ID": acc["anchor_id"], "Is_synthetic": 1,
            "Requirement_original": acc["requirement_en"], TEXT_COL: acc["es"],
            "Translation_source": "deepl+glossary_provisional",
            "Security": 1, **{c: acc["labels"].str.split("|").map(lambda s, c=c: int(c in s)) for c in SUBCATEGORY_COLS},
            "Prompt_version": "prompt_gen_v1", "Domain": acc["domain"], "Gen_ID": acc["gen_id"],
            "Filter_status": "accepted", "Label_verified_by": "SON-V_x3_provisional",
            "Sim_source": acc["sim_source"].round(4), "Sim_max_real": acc["sim_max_real"].round(4),
        })
        # Integrity of the anchors: every anchor is a real security requirement with those labels.
        for r in out.itertuples():
            anchor = df.loc[r.Anchor_ID]
            assert anchor["Security"] == 1 and all(anchor[c] == getattr(r, c) for c in SUBCATEGORY_COLS)
        out.to_csv(A / f"synthetic_{dataset}.csv", index=False, encoding="utf-8-sig")
        summary["by_dataset"][dataset] = {
            "generated": len(part), "accepted": len(out),
            "rejected_by_reason": part.loc[part["reject_reason"].ne(""), "reject_reason"].value_counts().to_dict(),
            "accepted_positives": {c: int(out[c].sum()) for c in SUBCATEGORY_COLS},
            "target_positives": {r.label: int(r.n_aug_target) for r in
                                 pd.read_csv(A / "budget.csv").query("dataset == @dataset").itertuples()},
        }
    (A / "filter_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
