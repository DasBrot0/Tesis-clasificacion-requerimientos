#!/usr/bin/env python3
"""G9: diversidad de los sintéticos aceptados (similitud coseno LaBSE media), por dataset.

    python scripts/augmentation/g9_diversity.py
    python scripts/build_report.py --condition c1_augmented      # la lleva al Excel

Para cada dataset, en español (el texto con que se entrena):
- sintético-sintético: media de la similitud entre cada par de sintéticos;
- real-real: lo mismo entre los requisitos reales de seguridad, como referencia;
- sintético-real: media entre cada sintético y cada requisito real de seguridad.
Si los sintéticos son tan variados como los reales, sintético-sintético queda cerca de real-real;
un valor mucho mayor indica variantes repetitivas.

Salida: outputs/output_raw/synthetic_diversity_labse.csv. Necesita transformers y descarga
sentence-transformers/LaBSE, igual que g5_g7_filter_build.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_g7_filter_build import labse  # noqa: E402
from src.config import AUGMENTATION_DIR, DATASET_NAMES, DATASETS, OUTPUT_RAW, TEXT_COL  # noqa: E402


def mean_pairwise(e: np.ndarray) -> float:
    sim = e @ e.T
    return float(sim[np.triu_indices(len(e), k=1)].mean())


def main() -> None:
    rows = []
    for dataset, path in DATASETS.items():
        synthetic = pd.read_csv(AUGMENTATION_DIR / f"synthetic_{dataset}.csv", encoding="utf-8-sig")
        real = pd.read_csv(path, encoding="utf-8-sig")
        real = real[real["Security"] == 1]
        e_syn = labse(synthetic[TEXT_COL].tolist()).numpy()
        e_real = labse(real[TEXT_COL].tolist()).numpy()
        rows.append({
            "dataset": DATASET_NAMES[dataset], "real_security": len(real),
            "sim_synthetic_synthetic": mean_pairwise(e_syn),
            "sim_real_real": mean_pairwise(e_real),
            "sim_synthetic_real": float((e_syn @ e_real.T).mean()),
        })
    table = pd.DataFrame(rows)
    OUTPUT_RAW.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_RAW / "synthetic_diversity_labse.csv", index=False, encoding="utf-8-sig")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
