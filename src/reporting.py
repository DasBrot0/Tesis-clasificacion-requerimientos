"""Reports built from finished runs: confusion matrices, comparison table and the Excel workbook.

Nothing here trains or imports torch: every file is rebuilt from what a run already saved
(summary.json and pooled_outer_test_predictions.csv), so `scripts/build_report.py` can refresh the
reports of finished runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS, DATASET_NAMES, DATASETS, LABEL_COLS, MODEL_DISPLAY, MODELS, run_dir

CONFUSION_CSV = "pooled_confusion_by_label.csv"
CONFUSION_PNG = "pooled_confusion_matrices.png"

# Spanish column titles for the workbook; the order is the order of the columns.
METRIC_TITLES = {
    "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1", "micro_f1": "Micro-F1",
    "macro_precision": "Macro-precisión", "macro_recall": "Macro-recall",
    "security_recall": "Recall Security", "subset_accuracy": "Exactitud de subconjunto",
    "hamming_loss": "Hamming loss",
}


# --------------------------------------------------------------------------- confusion matrices

def confusion_by_label(pooled: pd.DataFrame) -> pd.DataFrame:
    """One 2x2 matrix per label (TP, FP, FN, TN) from the pooled outer test predictions."""
    rows = []
    for label in LABEL_COLS:
        true = pooled[f"{label}_true"].astype(int)
        pred = pooled[f"{label}_pred"].astype(int)
        rows.append({
            "label": label,
            "TP": int(((true == 1) & (pred == 1)).sum()), "FP": int(((true == 0) & (pred == 1)).sum()),
            "FN": int(((true == 1) & (pred == 0)).sum()), "TN": int(((true == 0) & (pred == 0)).sum()),
        })
    return pd.DataFrame(rows)


def save_confusion_figure(confusion: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(confusion), figsize=(3.2 * len(confusion), 3.4))
    for ax, row in zip(axes, confusion.itertuples()):
        matrix = [[row.TN, row.FP], [row.FN, row.TP]]
        ax.imshow(matrix, cmap="Blues")
        peak = max(max(r) for r in matrix)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=12,
                        color="white" if matrix[i][j] > peak / 2 else "black")
        ax.set(title=row.label, xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"],
               yticklabels=["0", "1"], xlabel="predicho", ylabel="real")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_run_confusion(out_dir: Path, title: str) -> pd.DataFrame:
    pooled = pd.read_csv(out_dir / "pooled_outer_test_predictions.csv", encoding="utf-8-sig", dtype={"ID": str})
    confusion = confusion_by_label(pooled)
    confusion.to_csv(out_dir / CONFUSION_CSV, index=False, encoding="utf-8-sig")
    save_confusion_figure(confusion, out_dir / CONFUSION_PNG, title)
    return confusion


# --------------------------------------------------------------------------- finished runs

def finished_runs(condition: str) -> list[tuple[str, str, dict]]:
    """(dataset, model, summary) of every finished run of a condition, in the canonical order."""
    runs = []
    for dataset in DATASETS:
        for model_name in MODELS:
            path = run_dir(condition, dataset, model_name) / "summary.json"
            if path.exists():
                runs.append((dataset, model_name, json.loads(path.read_text(encoding="utf-8"))))
    return runs


def run_confusion(condition: str, dataset: str, model_name: str) -> pd.DataFrame:
    out_dir = run_dir(condition, dataset, model_name)
    if (out_dir / CONFUSION_CSV).exists():
        return pd.read_csv(out_dir / CONFUSION_CSV, encoding="utf-8-sig")
    return save_run_confusion(out_dir, f"{DATASET_NAMES[dataset]} — {MODEL_DISPLAY[model_name]} ({condition})")


def update_comparison_table(condition: str) -> pd.DataFrame:
    rows = []
    for dataset, model_name, report in finished_runs(condition):
        outer, pooled = report["outer_aggregate"], report["pooled_outer_test"]
        fmt = lambda name: f"{outer[name]['mean']:.4f} ± {outer[name]['std']:.4f}"
        rows.append({
            "Condición": condition, "Dataset": DATASET_NAMES[dataset], "Modelo": MODEL_DISPLAY[model_name],
            "Macro-F1 (5 folds externos)": fmt("macro_f1"), "Weighted-F1 (5 folds externos)": fmt("weighted_f1"),
            "Recall Security (5 folds externos)": fmt("security_recall"),
            "Macro-F1 agrupado": pooled["macro_f1"], "Weighted-F1 agrupado": pooled["weighted_f1"],
            "Micro-F1 agrupado": pooled["micro_f1"], "Recall Security agrupado": pooled["security_recall"],
            **{f"F1 {row['label']}": row["f1"] for row in report["pooled_outer_test_per_label"]},
            "Etiquetas en el promedio": ", ".join(report["labels_in_macro_average"]),
            "Sintéticos": report["synthetic_rows"], "Minutos": report["elapsed_minutes"],
        })
    table = pd.DataFrame(rows)
    path = ARTIFACTS / condition / "model_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return table


# --------------------------------------------------------------------------- Excel workbook

def export_excel(condition: str) -> Path | None:
    """artifacts/<condition>/results_<condition>.xlsx: general metrics, metrics by class and
    confusion matrices of every finished run, formatted to be read as is."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    runs = finished_runs(condition)
    if not runs:
        return None

    header_fill = PatternFill("solid", fgColor="1F4E78")
    group_fill = PatternFill("solid", fgColor="D9E1F2")
    hit_fill = PatternFill("solid", fgColor="E2EFDA")
    miss_fill = PatternFill("solid", fgColor="FCE4D6")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    score_scale = dict(start_type="num", start_value=0, start_color="F8696B",
                       mid_type="num", mid_value=0.5, mid_color="FFEB84",
                       end_type="num", end_value=1, end_color="63BE7B")

    def write_table(ws, top: int, headers: list[str], rows: list[list], formats: dict[int, str]) -> int:
        """Header + rows starting at `top`; returns the last row written."""
        for col, title in enumerate(headers, start=1):
            cell = ws.cell(top, col, title)
            cell.fill, cell.font, cell.alignment, cell.border = header_fill, header_font, center, border
        for r, values in enumerate(rows, start=top + 1):
            for col, value in enumerate(values, start=1):
                cell = ws.cell(r, col, value)
                cell.border = border
                cell.alignment = Alignment(horizontal="left" if isinstance(value, str) else "center")
                if col in formats:
                    cell.number_format = formats[col]
        ws.row_dimensions[top].height = 32
        return top + len(rows)

    def bold_best(ws, first: int, last: int, col: int, key_col: int = 1, lower_is_better: bool = False) -> None:
        """Bold the best value of `col` within each block of rows that share the value of `key_col`."""
        blocks: dict[str, list[int]] = {}
        for r in range(first, last + 1):
            blocks.setdefault(ws.cell(r, key_col).value, []).append(r)
        for rows in blocks.values():
            pick = min if lower_is_better else max
            best = pick(rows, key=lambda r: ws.cell(r, col).value)
            ws.cell(best, col).font = Font(bold=True)

    def widths(ws, values: dict[int, float]) -> None:
        for col, width in values.items():
            ws.column_dimensions[get_column_letter(col)].width = width

    wb = Workbook()

    # 1) General metrics: pooled outer test + mean ± std of the five outer folds.
    ws = wb.active
    ws.title = "Resumen general"
    ws["A1"] = f"Métricas generales — condición {condition}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("Agrupado: predicciones de prueba de los 5 folds externos juntas (cada requisito real "
                "una vez). Folds: media ± desviación estándar de los 5 folds externos. "
                "En negrita, el mejor modelo de cada dataset.")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:P2")
    ws.row_dimensions[2].height = 32
    names = list(METRIC_TITLES)
    ws.cell(4, 3, "Agrupado (5 folds externos)")
    ws.merge_cells(start_row=4, start_column=3, end_row=4, end_column=2 + len(names))
    ws.cell(4, 3 + len(names), "Media ± desv. de los 5 folds externos")
    ws.merge_cells(start_row=4, start_column=3 + len(names), end_row=4, end_column=2 + 2 * len(names))
    for col in (3, 3 + len(names)):
        ws.cell(4, col).fill, ws.cell(4, col).font, ws.cell(4, col).alignment = group_fill, Font(bold=True), center
    headers = ["Dataset", "Modelo", *METRIC_TITLES.values(), *METRIC_TITLES.values()]
    rows = []
    for dataset, model_name, report in runs:
        pooled, outer = report["pooled_outer_test"], report["outer_aggregate"]
        rows.append([DATASET_NAMES[dataset], MODEL_DISPLAY[model_name], *(pooled[n] for n in names),
                     *(f"{outer[n]['mean']:.3f} ± {outer[n]['std']:.3f}" for n in names)])
    last = write_table(ws, 5, headers, rows, {c: "0.000" for c in range(3, 3 + len(names))})
    for offset, name in enumerate(names):
        bold_best(ws, 6, last, 3 + offset, lower_is_better=name == "hamming_loss")
    # Hamming loss (last pooled column) is better when lower: it stays out of the color scale.
    ws.conditional_formatting.add(f"C6:{get_column_letter(2 + len(names) - 1)}{last}", ColorScaleRule(**score_scale))
    widths(ws, {1: 14, 2: 14, **{c: 13 for c in range(3, 3 + 2 * len(names))}})
    ws.freeze_panes = "C6"

    # 2) F1 by class, one row per (dataset, model): the compact view to compare models.
    ws = wb.create_sheet("F1 por clase")
    ws["A1"] = "F1 por clase (predicciones externas agrupadas)"
    ws["A1"].font = Font(bold=True, size=14)
    headers = ["Dataset", "Modelo", *LABEL_COLS, "Macro-F1"]
    rows = []
    for dataset, model_name, report in runs:
        f1 = {row["label"]: row["f1"] for row in report["pooled_outer_test_per_label"]}
        rows.append([DATASET_NAMES[dataset], MODEL_DISPLAY[model_name], *(f1[l] for l in LABEL_COLS),
                     report["pooled_outer_test"]["macro_f1"]])
    last = write_table(ws, 3, headers, rows, {c: "0.000" for c in range(3, 4 + len(LABEL_COLS))})
    for col in range(3, 4 + len(LABEL_COLS)):
        bold_best(ws, 4, last, col)
    ws.conditional_formatting.add(f"C4:{get_column_letter(3 + len(LABEL_COLS))}{last}", ColorScaleRule(**score_scale))
    widths(ws, {1: 14, 2: 14, **{c: 16 for c in range(3, 4 + len(LABEL_COLS))}})
    ws.freeze_panes = "C4"

    # 3) Every metric by class, with the confusion counts behind it.
    ws = wb.create_sheet("Métricas por clase")
    ws["A1"] = "Métricas por clase (predicciones externas agrupadas)"
    ws["A1"].font = Font(bold=True, size=14)
    headers = ["Dataset", "Modelo", "Clase", "Precisión", "Recall", "F1", "Soporte",
               "VP", "FP", "FN", "VN", "En el promedio"]
    rows = []
    for dataset, model_name, report in runs:
        confusion = run_confusion(condition, dataset, model_name).set_index("label")
        for row in report["pooled_outer_test_per_label"]:
            counts = confusion.loc[row["label"]]
            rows.append([DATASET_NAMES[dataset], MODEL_DISPLAY[model_name], row["label"], row["precision"],
                         row["recall"], row["f1"], int(row["support"]), int(counts["TP"]), int(counts["FP"]),
                         int(counts["FN"]), int(counts["TN"]), "Sí" if row["in_macro_average"] else "No"])
    last = write_table(ws, 3, headers, rows, {4: "0.000", 5: "0.000", 6: "0.000"})
    ws.conditional_formatting.add(f"D4:F{last}", ColorScaleRule(**score_scale))
    ws.auto_filter.ref = f"A3:{get_column_letter(len(headers))}{last}"
    widths(ws, {1: 14, 2: 14, 3: 17, **{c: 11 for c in range(4, len(headers) + 1)}, len(headers): 15})
    ws.freeze_panes = "D4"

    # 4) Confusion matrices: one block per run, one 2x2 matrix per class.
    ws = wb.create_sheet("Matrices de confusión")
    ws["A1"] = "Matrices de confusión por clase (predicciones externas agrupadas)"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = "Filas: valor real; columnas: valor predicho. Verde: aciertos (VN, VP); naranja: errores (FP, FN)."
    top = 4
    for dataset, model_name, _ in runs:
        confusion = run_confusion(condition, dataset, model_name)
        ws.cell(top, 1, f"{DATASET_NAMES[dataset]} — {MODEL_DISPLAY[model_name]}").font = Font(bold=True, size=12)
        for k, row in enumerate(confusion.itertuples()):
            left = 1 + 4 * k
            ws.cell(top + 1, left, row.label)
            ws.merge_cells(start_row=top + 1, start_column=left, end_row=top + 1, end_column=left + 2)
            title = ws.cell(top + 1, left)
            title.fill, title.font, title.alignment = header_fill, header_font, center
            cells = [["Real \\ Pred.", "0", "1"], ["0", row.TN, row.FP], ["1", row.FN, row.TP]]
            for i, values in enumerate(cells):
                for j, value in enumerate(values):
                    cell = ws.cell(top + 2 + i, left + j, value)
                    cell.border, cell.alignment = border, center
                    if i == 0 or j == 0:
                        cell.fill, cell.font = group_fill, Font(bold=True)
                    else:
                        cell.fill = hit_fill if i == j else miss_fill
        top += 6
    widths(ws, {c: (12 if c % 4 == 1 else 9 if c % 4 else 3) for c in range(1, 4 * len(LABEL_COLS) + 1)})

    # 5) Each outer fold, for the variability behind the mean ± std.
    ws = wb.create_sheet("Folds externos")
    ws["A1"] = "Métricas de cada fold externo"
    ws["A1"].font = Font(bold=True, size=14)
    headers = ["Dataset", "Modelo", "Fold externo", *METRIC_TITLES.values()]
    rows = [[DATASET_NAMES[dataset], MODEL_DISPLAY[model_name], fold["outer"] + 1,
             *(fold["test"][n] for n in names)]
            for dataset, model_name, report in runs for fold in report["outer_folds"]]
    last = write_table(ws, 3, headers, rows, {c: "0.000" for c in range(4, 4 + len(names))})
    ws.auto_filter.ref = f"A3:{get_column_letter(len(headers))}{last}"
    widths(ws, {1: 14, 2: 14, 3: 11, **{c: 13 for c in range(4, 4 + len(names))}})
    ws.freeze_panes = "D4"

    path = ARTIFACTS / condition / f"results_{condition}.xlsx"
    wb.save(path)
    return path
