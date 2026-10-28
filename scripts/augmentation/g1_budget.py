#!/usr/bin/env python3
"""G1 (PLAN_AUMENTO_DATOS.md): presupuesto por subcategoría, fuentes y lotes para SON-G.

    python scripts/augmentation/g1_budget.py

Con CV anidada todos los requisitos reales pueden ser fuente (cada uno es de prueba en un solo
fold externo; G7 excluye sus variantes de ese fold).

    n_aug_c = min(2 * n_c, n_max - n_c)     (n_max: subcategoría más frecuente del dataset)

Asignación voraz: cada variante hereda todas las subcategorías de su fuente, así que una fuente
con dos subcategorías descuenta de ambas. Se prefieren las fuentes cuyas subcategorías todavía
faltan todas y, entre ellas, las de mayor peso de clase (Aldhafer et al., 2026, ec. 6). Máximo
MAX_PER_SOURCE variantes aceptadas por fuente; se piden OVERGEN veces más (para G5 y G6).

Salidas en inputs/interim/augmentation/:
    budget.csv            presupuesto por dataset y subcategoría
    sources.csv           una fila por fuente asignada
    gen_batches/batch_XX.jsonl  entrada de SON-G
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    AUGMENTATION_DIR, DATASETS, MIN_POSITIVES_PER_LABEL, SEED, SUBCATEGORY_COLS,
)

MAX_PER_SOURCE = 2
OVERGEN = 1.5
BATCH = 12
# Los 10 tipos de software de Pham et al. (2025) y 4 dominios de los propios datasets
# (tarjeta inteligente, salud, banca, hogar conectado) [Decisión propia].
DOMAINS = [
    "web browser", "database server", "air traffic control software", "simulation software",
    "payroll and human resources management system", "compiler", "data warehousing system",
    "online booking system", "sales forecasting system", "screen capture tool",
    "smart-card electronic purse", "hospital patient-record system", "online banking application",
    "smart-home IoT controller",
]


def allocate(df: pd.DataFrame, labels: list[str], need: dict[str, int], rng) -> dict[str, int]:
    sources = df[df["Security"].eq(1) & df[labels].any(axis=1)].copy()
    n = df[labels].sum()
    w = n / n.sum()
    sources["w_class"] = sources[labels].apply(
        lambda r: (1 / w[r.astype(bool)].sum()) / r.sum(), axis=1)
    sources["tiebreak"] = rng.permutation(len(sources))
    used = dict.fromkeys(sources["ID"], 0)
    need = dict(need)
    while any(v > 0 for v in need.values()):
        best, best_key = None, None
        for r in sources.itertuples():
            if used[r.ID] >= MAX_PER_SOURCE:
                continue
            own = [c for c in labels if getattr(r, c)]
            missing = sum(need[c] > 0 for c in own)
            if not missing:
                continue
            key = (missing / len(own), r.w_class, -used[r.ID], -r.tiebreak)
            if best_key is None or key > best_key:
                best, best_key = r, key
        if best is None:
            break
        used[best.ID] += 1
        for c in labels:
            if getattr(best, c):
                need[c] -= 1
    return {k: v for k, v in used.items() if v}


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = AUGMENTATION_DIR
    (out / "gen_batches").mkdir(parents=True, exist_ok=True)
    budget_rows, source_rows = [], []
    for dataset, path in DATASETS.items():
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})
        labels = [c for c in SUBCATEGORY_COLS if df[c].sum() >= MIN_POSITIVES_PER_LABEL]
        n = {c: int(df[c].sum()) for c in labels}
        n_max = max(n.values())
        need = {c: min(2 * n[c], n_max - n[c]) for c in labels}
        used = allocate(df, labels, need, rng)
        chosen = df[df["ID"].isin(used)]
        achieved = {c: int(sum(used[i] for i in chosen.loc[chosen[c].eq(1), "ID"])) for c in labels}
        for c in labels:
            budget_rows.append({"dataset": dataset, "label": c, "n_real": n[c], "n_max": n_max,
                                "n_aug_target": need[c], "n_aug_assigned": achieved[c]})
        for r in chosen.itertuples():
            k_accept = used[r.ID]
            k_request = math.ceil(k_accept * OVERGEN)
            source_rows.append({
                "dataset": dataset, "anchor_id": r.ID, "labels": "|".join(c for c in labels if getattr(r, c)),
                "k_accept": k_accept, "k_request": k_request,
                "domains": "|".join(rng.choice(DOMAINS, size=k_request, replace=False)),
                "text_en": r.Requirement_original,
            })
    budget = pd.DataFrame(budget_rows)
    sources = pd.DataFrame(source_rows)
    budget.to_csv(out / "budget.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(out / "sources.csv", index=False, encoding="utf-8-sig")
    for old in (out / "gen_batches").glob("batch_*.jsonl"):
        old.unlink()
    for i in range(0, len(sources), BATCH):
        with open(out / "gen_batches" / f"batch_{i // BATCH + 1:02d}.jsonl", "w", encoding="utf-8") as fh:
            for r in sources.iloc[i:i + BATCH].itertuples():
                fh.write(json.dumps({"anchor_id": r.anchor_id, "source_requirement": r.text_en,
                                     "labels": r.labels.split("|"), "domains": r.domains.split("|")},
                                    ensure_ascii=False) + "\n")
    print(budget.to_string(index=False))
    print(sources.groupby("dataset")[["k_accept", "k_request"]].sum())
    print(f"{len(sources)} fuentes, {sources.k_request.sum()} variantes a generar, "
          f"{math.ceil(len(sources) / BATCH)} lotes")


if __name__ == "__main__":
    main()
