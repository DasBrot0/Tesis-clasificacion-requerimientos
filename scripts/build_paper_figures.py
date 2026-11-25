#!/usr/bin/env python3
"""Figuras de la sección de resultados del paper, a partir de las corridas ya terminadas.

    python scripts/build_paper_figures.py                                   # corrida actual de C1
    python scripts/build_paper_figures.py --root old/run_2026-10-27_c1_pre_dedup/output_raw/c1_augmented

Lee pooled_outer_test_predictions.csv y summary.json de cada <dataset>/<modelo>/, más best_model.csv y
subcategory_confusion.csv de la condición (los escribe build_report.py), y guarda en --out:
- fig_conteos.png: positivos de cada etiqueta por dataset (barras).
- fig_confusion_seguridad.png: matriz de confusión de Security del mejor modelo de cada dataset.
- fig_roc_seguridad.png: curvas ROC de Security de todos los modelos, con su AUC.
- fig_roc_subcategorias.png: curvas ROC de cada subcategoría (fila = dataset, columna = subcategoría).
- fig_confusion_subcategorias.png: confusión entre subcategorías del mejor modelo de cada dataset.
Las curvas ROC usan las probabilidades de prueba de las cinco particiones externas, antes del umbral.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATASETS, LABEL_COLS, MODEL_DISPLAY, MODELS, OUTPUT_RAW, REPORTS  # noqa: E402

DATASET_TITLES = {"promise": "PROMISE_exp_translated", "secreq": "SecReq", "dosspre": "DOSSPRE"}
LABELS_ES = {"Security": "Seguridad", "Confidentiality": "Confidencialidad", "Integrity": "Integridad",
             "Authenticity": "Autenticidad", "Resistance": "Resistencia", "Ninguna": "Ninguna"}
# The abbreviations of Tabla 5.3.
LABELS_SHORT = {"Security": "SEG", "Confidentiality": "CON", "Integrity": "INT", "Authenticity": "AUT",
                "Resistance": "RES", "Ninguna": "Ninguna"}
SHORT_MODEL = {"cnn": "CNN", "bilstm": "BiLSTM", "beto": "BETO", "roberta_bne": "RoB.", "distilbeto": "Dist."}
# The color follows the model in every figure (categorical slots in fixed order, validated for CVD);
# every curve is a solid line, so the color alone identifies the model.
MODEL_COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MODEL_LINES = dict.fromkeys(MODELS, "-")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.titlesize": 10, "legend.frameon": True, "legend.framealpha": 0.9,
                     "legend.edgecolor": "none"})
DECIMAL_COMMA = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))


def comma(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def load_runs(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    runs = {}
    for dataset in DATASETS:
        for model in MODELS:
            path = root / dataset / model / "pooled_outer_test_predictions.csv"
            if path.exists():
                runs.setdefault(dataset, {})[model] = pd.read_csv(path, encoding="utf-8-sig")
    return runs


def best_models(root: Path) -> dict[str, str]:
    key = {display: name for name, display in MODEL_DISPLAY.items()}
    titles = {"PROMISE_exp": "promise", "SecReq": "secreq", "DOSSPRE": "dosspre"}
    table = pd.read_csv(root / "best_model.csv", encoding="utf-8-sig")
    return {titles[row.dataset]: key[row.best_model] for row in table.itertuples()}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.set_major_formatter(DECIMAL_COMMA)
    ax.yaxis.set_major_formatter(DECIMAL_COMMA)


def fig_counts(root: Path, runs, out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.3, 2.4), sharey=True)
    for ax, (dataset, models) in zip(axes, runs.items()):
        frame = next(iter(models.values()))
        counts = [int(frame[f"{label}_true"].sum()) for label in LABEL_COLS]
        summary = json.loads((root / dataset / next(iter(models)) / "summary.json").read_text(encoding="utf-8"))
        style(ax)  # before set_xticks: its number formatter would replace the label names
        bars = ax.bar(range(len(LABEL_COLS)), counts, width=0.7, color=MODEL_COLORS[MODELS[0]])
        ax.bar_label(bars, labels=[str(c) for c in counts], padding=2, fontsize=8, color=INK)
        ax.set_xticks(range(len(LABEL_COLS)), [LABELS_SHORT[label] for label in LABEL_COLS], fontsize=8)
        ax.set_title(f"{DATASET_TITLES[dataset]}\n{len(frame)} reales; +{summary['synthetic_rows']} sintéticos",
                     fontsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Requerimientos positivos")
    fig.tight_layout()
    fig.savefig(out / "fig_conteos.png", dpi=300)
    plt.close(fig)


def draw_matrix(ax, matrix: np.ndarray, rows: list[str], cols: list[str], percent: bool,
                masked: tuple[int, int] | None = None) -> None:
    shown = matrix.astype(float).copy()
    if masked is not None:
        shown[masked] = np.nan
    scale = shown / shown.sum(axis=1, keepdims=True) if percent else shown
    ax.imshow(np.ma.masked_invalid(scale), cmap="Blues", vmin=0, vmax=np.nanmax(scale))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = str(int(matrix[i, j]))
            if percent:
                text += f"\n({comma(100 * matrix[i, j] / matrix[i].sum(), 1)} %)"
            dark = not np.isnan(scale[i, j]) and scale[i, j] > 0.6 * np.nanmax(scale)
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_security_confusion(runs, best, out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.3, 2.4))
    for ax, dataset in zip(axes, runs):
        frame = runs[dataset][best[dataset]]
        y, p = frame["Security_true"], frame["Security_pred"]
        matrix = np.array([[((y == 0) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum()],
                           [((y == 1) & (p == 0)).sum(), ((y == 1) & (p == 1)).sum()]])
        draw_matrix(ax, matrix, ["No seg.", "Seg."], ["No seg.", "Seg."], percent=True)
        ax.set_title(f"{DATASET_TITLES[dataset]}\n({MODEL_DISPLAY[best[dataset]]})", fontsize=9)
        ax.set_xlabel("Predicho", fontsize=8)
    axes[0].set_ylabel("Real", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_seguridad.png", dpi=300)
    plt.close(fig)


def roc_panel(ax, models: dict[str, pd.DataFrame], label: str, legend: bool) -> None:
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
    for model, frame in models.items():
        y, score = frame[f"{label}_true"], frame[f"{label}_prob"]
        fpr, tpr, _ = roc_curve(y, score)
        name = f"{MODEL_DISPLAY[model]} ({comma(roc_auc_score(y, score))})" if legend else MODEL_DISPLAY[model]
        ax.plot(fpr, tpr, color=MODEL_COLORS[model], linestyle=MODEL_LINES[model], linewidth=1.5, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    style(ax)


def fig_security_roc(runs, out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.3, 2.9))
    for ax, (dataset, models) in zip(axes, runs.items()):
        roc_panel(ax, models, "Security", legend=True)
        ax.set_title(DATASET_TITLES[dataset], fontsize=9)
        ax.set_xlabel("Tasa de falsos positivos", fontsize=8)
        ax.legend(loc="lower right", fontsize=7, title="Modelo (AUC)", title_fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Tasa de verdaderos positivos", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_roc_seguridad.png", dpi=300)
    plt.close(fig)


def fig_subcategory_roc(runs, out: Path) -> None:
    subcategories = LABEL_COLS[1:]
    fig, axes = plt.subplots(len(runs), len(subcategories), figsize=(6.3, 5.2), sharex=True, sharey=True)
    for r, (dataset, models) in enumerate(runs.items()):
        for c, label in enumerate(subcategories):
            ax = axes[r, c]
            roc_panel(ax, models, label, legend=False)
            aucs = [f"{SHORT_MODEL[m]} {comma(roc_auc_score(f[f'{label}_true'], f[f'{label}_prob']), 2)}"
                    for m, f in models.items()]
            ax.text(0.97, 0.03, "\n".join(aucs), transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                    color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(LABELS_ES[label], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{DATASET_TITLES[dataset]}\nTVP", fontsize=7)
            if r == len(runs) - 1:
                ax.set_xlabel("TFP", fontsize=7)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig_roc_subcategorias.png", dpi=300)
    plt.close(fig)


def fig_subcategory_confusion(root: Path, runs, best, out: Path) -> None:
    table = pd.read_csv(root / "subcategory_confusion.csv", encoding="utf-8-sig")
    titles = {"promise": "PROMISE_exp", "secreq": "SecReq", "dosspre": "DOSSPRE"}
    order = ["Confidentiality", "Integrity", "Authenticity", "Resistance", "Ninguna"]
    names = [LABELS_SHORT[label] for label in order]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.3, 2.6))
    for ax, dataset in zip(axes, runs):
        part = table[(table["dataset"] == titles[dataset]) & (table["model"] == MODEL_DISPLAY[best[dataset]])]
        matrix = part.pivot(index="real", columns="predicted", values="count").reindex(index=order, columns=order)
        # The no-subcategory/no-subcategory cell holds most requirements; it is printed but not colored.
        draw_matrix(ax, matrix.to_numpy(), names, [*names[:-1], "Ning."], percent=False, masked=(4, 4))
        ax.set_title(f"{DATASET_TITLES[dataset]}\n({MODEL_DISPLAY[best[dataset]]})", fontsize=9)
        ax.set_xlabel("Predicha", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Real", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_subcategorias.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=OUTPUT_RAW / "c1_augmented",
                        help="Carpeta de la condición con <dataset>/<modelo>/ y best_model.csv.")
    parser.add_argument("--out", type=Path, default=REPORTS / "c1_augmented" / "paper_figures")
    parser.add_argument("--extra-root", type=Path, default=None,
                        help="Otra corrida de la que se toman solo los modelos que faltan en --root (curvas ROC).")
    args = parser.parse_args()
    runs = load_runs(args.root)
    if args.extra_root is not None:
        for dataset, models in load_runs(args.extra_root).items():
            for model, frame in models.items():
                runs.setdefault(dataset, {}).setdefault(model, frame)
        # Keep the canonical model order so every figure lists the models the same way.
        runs = {d: {m: runs[d][m] for m in MODELS if m in runs[d]} for d in runs}
    best = best_models(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_counts(args.root, runs, args.out)
    fig_security_confusion(runs, best, args.out)
    fig_security_roc(runs, args.out)
    fig_subcategory_roc(runs, args.out)
    fig_subcategory_confusion(args.root, runs, best, args.out)
    print(f"Figuras en {args.out}")


if __name__ == "__main__":
    main()
