"""Statistical tests used to compare models.

Non-parametric Scott-Knott ESD, as described by Binkhonain & Alfayez (2025, §3.7):

1. Partition: the treatments (models) are sorted by median; among the cut points of that order,
   the one with the largest Kruskal-Wallis H between the two sides is chosen. If that split is
   significant (p < alpha), each side is partitioned again; otherwise the set is one group.
2. Effect size: adjacent groups whose difference is negligible by Cliff's delta are merged, so
   every surviving split is significant and non-negligible. The paper does not give the cut-off;
   |d| < 0.147 is the one the ScottKnottESD R package applies to call an effect negligible.

Each treatment contributes one observation per outer fold (its macro-F1 on that fold). The test
treats them as independent samples, as Scott-Knott ESD does.

Cochran's Q and exact McNemar with Holm's correction compare the hits of several models on the
same requirements (Limaylla-Lunarejo et al., 2026; Shafikuzzaman et al., 2025).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, kruskal

NEGLIGIBLE_DELTA = 0.147


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over every pair of observations."""
    diff = np.subtract.outer(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float((np.sign(diff)).mean())


def delta_magnitude(delta: float) -> str:
    size = abs(delta)
    if size < NEGLIGIBLE_DELTA:
        return "despreciable"
    if size < 0.33:
        return "pequeño"
    if size < 0.474:
        return "mediano"
    return "grande"


def _kruskal_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        statistic, p_value = kruskal(a, b)
    except ValueError:  # every observation identical: no evidence of a difference
        return 0.0, 1.0
    return float(statistic), float(p_value)


def _partition(names: list[str], samples: dict[str, np.ndarray], alpha: float) -> list[list[str]]:
    if len(names) < 2:
        return [names]
    best = None
    for cut in range(1, len(names)):
        left = np.concatenate([samples[n] for n in names[:cut]])
        right = np.concatenate([samples[n] for n in names[cut:]])
        statistic, p_value = _kruskal_p(left, right)
        if best is None or statistic > best[0]:
            best = (statistic, p_value, cut)
    _, p_value, cut = best
    if p_value >= alpha:
        return [names]
    return _partition(names[:cut], samples, alpha) + _partition(names[cut:], samples, alpha)


def scott_knott_esd(samples: dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """Rank (1 = best) and group of each treatment; treatments sharing a rank do not differ."""
    samples = {name: np.asarray(values, dtype=float) for name, values in samples.items()}
    order = sorted(samples, key=lambda n: (np.median(samples[n]), np.mean(samples[n])), reverse=True)
    groups = _partition(order, samples, alpha)

    # Merge adjacent groups until every remaining difference is non-negligible.
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for i in range(len(groups) - 1):
            upper = np.concatenate([samples[n] for n in groups[i]])
            lower = np.concatenate([samples[n] for n in groups[i + 1]])
            if abs(cliffs_delta(upper, lower)) < NEGLIGIBLE_DELTA:
                groups[i:i + 2] = [groups[i] + groups[i + 1]]
                merged = True
                break

    rows = []
    for rank, group in enumerate(groups, start=1):
        following = groups[rank] if rank < len(groups) else None
        if following:
            delta = cliffs_delta(np.concatenate([samples[n] for n in group]),
                                 np.concatenate([samples[n] for n in following]))
            _, p_value = _kruskal_p(np.concatenate([samples[n] for n in group]),
                                    np.concatenate([samples[n] for n in following]))
        for name in group:
            values = samples[name]
            rows.append({
                "treatment": name, "rank": rank, "group": " = ".join(group),
                "median": float(np.median(values)), "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "n": len(values),
                "delta_vs_next_group": delta if following else np.nan,
                "magnitude_vs_next_group": delta_magnitude(delta) if following else "",
                "p_vs_next_group": p_value if following else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- paired hits

def cochran_q(hits: np.ndarray) -> tuple[float, float]:
    """Cochran's Q over an (n requirements, k models) 0/1 matrix of hits; returns (Q, p)."""
    hits = np.asarray(hits, dtype=int)
    k = hits.shape[1]
    columns, rows = hits.sum(axis=0), hits.sum(axis=1)
    total = rows.sum()
    denominator = k * total - (rows ** 2).sum()
    if denominator == 0:  # every requirement is a hit for all models or for none
        return 0.0, 1.0
    q = (k - 1) * (k * (columns ** 2).sum() - total ** 2) / denominator
    return float(q), float(chi2.sf(q, k - 1))


def mcnemar_exact(hits_a: np.ndarray, hits_b: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar: (only A right, only B right, p)."""
    hits_a, hits_b = np.asarray(hits_a, dtype=bool), np.asarray(hits_b, dtype=bool)
    only_a, only_b = int((hits_a & ~hits_b).sum()), int((~hits_a & hits_b).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p_value)


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
