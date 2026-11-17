"""Agreement between the two experts and the consensus sheet with only their disagreements (E3, E4).

Reads the filled sheets 1_Piloto and 2_Muestra_ciegas of both workbooks, computes the agreement on the
independent answers (Cohen's kappa per label, percent agreement, exact match) and writes Consenso.xlsx.
In the consensus sheets the two answers appear as "A" and "B", shuffled per row, so the moderator (A1)
does not reveal whose answer is whose; the hidden column "A_es" keeps the mapping.

Usage: py -I scripts/expert_review/build_consensus_sheet.py <Validacion_EXP1.xlsx> <Validacion_EXP2.xlsx> [out_dir]
Output: Consenso.xlsx and agreement.csv in out_dir (default inputs/interim/expert_review/)
"""
import random
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "inputs" / "interim" / "expert_review"
SUBCATS = ["Confidentiality", "Integrity", "Authenticity", "Resistance"]
SHEETS = {"1_Piloto": "Piloto (E3, no cuenta para el κ)", "2_Muestra_ciegas": "Muestra a ciegas (E4)"}
FIRST = 5
FONT, NAVY = "Arial", "1F3864"
INPUT_FILL, REF_FILL = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def read_answers(path, sheet):
    """Columns of the blind sheets: B ID, C text, D Spanish, E-H subcategories, I none, J not security, K doubt, L comment."""
    ws = load_workbook(path, data_only=True)[sheet]
    rows = []
    for r in range(FIRST, ws.max_row + 1):
        if not ws[f"B{r}"].value:
            continue
        row = {"ID": ws[f"B{r}"].value, "text": ws[f"C{r}"].value, "es": ws[f"D{r}"].value}
        for col, label in zip("EFGH", SUBCATS):
            row[label] = int(ws[f"{col}{r}"].value == "Sí")
        row["none"] = int(ws[f"I{r}"].value == "Sí")
        row["not_security"] = int(ws[f"J{r}"].value == "Sí")
        row["answered"] = int(row["none"] == 1 or any(row[c] for c in SUBCATS))
        row["doubt"] = ws[f"K{r}"].value or ""
        row["comment"] = ws[f"L{r}"].value or ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("ID")


def kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def describe(row):
    labels = ", ".join(c for c in SUBCATS if row[c]) or ("Ninguna aplica" if row["none"] else "(sin responder)")
    extras = [labels]
    if row["not_security"]:
        extras.append("no le parece de seguridad")
    if row["doubt"]:
        extras.append(f"duda: {row['doubt']}")
    if row["comment"]:
        extras.append(f"comentario: {row['comment']}")
    return "; ".join(extras)


def write_sheet(wb, name, heading, frame):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws["A1"] = heading
    ws["A1"].font = Font(name=FONT, size=14, bold=True, color=NAVY)
    ws["A2"] = ("Solo los requisitos en que los expertos no coincidieron. Lean las dos respuestas, discútanlas con "
                "la guía y marquen la etiqueta acordada. Si no hay acuerdo, decide EXP1: marquen «Sin consenso».")
    ws["A2"].font = Font(name=FONT, size=10, italic=True, color="404040")
    cols = [("N°", 5), ("ID", 13), ("Requisito (inglés)", 58), ("Traducción al español (referencia)", 40),
            ("Respuesta A", 30), ("Respuesta B", 30), ("Difieren en", 18)] + [(c, 13) for c in SUBCATS] + [
           ("Ninguna aplica", 11), ("Sin consenso (decidió EXP1)", 12), ("Nota", 28), ("Estado", 11), ("A_es", 8)]
    for j, (head, width) in enumerate(cols, start=1):
        cell = ws.cell(row=4, column=j, value=head)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.column_dimensions["P"].hidden = True
    ws.row_dimensions[4].height = 42
    yesno = DataValidation(type="list", formula1='"Sí,No"', allow_blank=True)
    ws.add_data_validation(yesno)
    for k, row in enumerate(frame.itertuples(), start=1):
        r = FIRST + k - 1
        # H-K subcategories, L "Ninguna aplica", M "Sin consenso" (a flag only)
        status = (f'=IF(AND(L{r}="Sí",COUNTIF(H{r}:K{r},"Sí")>0),"Revisar",'
                  f'IF(OR(COUNTIF(H{r}:K{r},"Sí")>0,L{r}="Sí"),"Completo","Pendiente"))')
        values = [k, row.Index, row.text, row.es, row.answer_a, row.answer_b, row.differ,
                  None, None, None, None, None, None, None, status, row.a_is]
        for j, value in enumerate(values, start=1):
            cell = ws.cell(row=r, column=j, value=value)
            cell.font = Font(name=FONT, size=10)
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="top",
                                       horizontal="center" if 8 <= j <= 15 else None)
            if 8 <= j <= 14:
                cell.fill = INPUT_FILL
                if j <= 13:
                    yesno.add(cell)
            elif j in (4, 5, 6, 7):
                cell.fill = REF_FILL
        ws.row_dimensions[r].height = max(30, 14 * (len(str(row.text)) // 62 + 1) + 4)
    last = FIRST + len(frame) - 1
    if len(frame):
        green = PatternFill("solid", fgColor="C6EFCE")
        ws.conditional_formatting.add(f"H{FIRST}:M{last}", FormulaRule(formula=[f'H{FIRST}="Sí"'], fill=green))
        ws.conditional_formatting.add(f"O{FIRST}:O{last}", FormulaRule(formula=[f'O{FIRST}="Completo"'], fill=green))
        ws.conditional_formatting.add(f"O{FIRST}:O{last}", FormulaRule(
            formula=[f'O{FIRST}="Revisar"'], fill=PatternFill("solid", fgColor="F8CBAD")))
    ws.freeze_panes = "H5"


def main(path1, path2, out=OUT):
    out = Path(out)
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(42)
    wb = Workbook()
    summary = wb.active
    summary.title = "Acuerdo"
    summary.sheet_view.showGridLines = False
    summary["A1"] = "Acuerdo entre expertos antes de la reunión de consenso"
    summary["A1"].font = Font(name=FONT, size=14, bold=True, color=NAVY)
    summary["A2"] = ("Valores calculados por build_consensus_sheet.py con las respuestas independientes. "
                     "κ de Cohen por etiqueta; «exacto» = las cuatro subcaracterísticas iguales.")
    summary["A2"].font = Font(name=FONT, size=10, italic=True, color="404040")
    table = []
    for sheet, heading in SHEETS.items():
        a, b = read_answers(path1, sheet), read_answers(path2, sheet).reindex(read_answers(path1, sheet).index)
        both = (a.answered == 1) & (b.answered == 1)
        for label in SUBCATS + ["not_security"]:
            x, y = a.loc[both, label], b.loc[both, label]
            table.append({"Hoja": sheet, "Etiqueta": "No le parece de seguridad" if label == "not_security" else label, "Respondidos por ambos": int(both.sum()),
                          "Positivos EXP1": int(x.sum()), "Positivos EXP2": int(y.sum()),
                          "Acuerdo %": round(100 * (x == y).mean(), 1) if both.any() else None,
                          "κ de Cohen": round(kappa(x, y), 3) if both.any() else None})
        exact = (a.loc[both, SUBCATS] == b.loc[both, SUBCATS]).all(axis=1)
        table.append({"Hoja": sheet, "Etiqueta": "Exacto (4 subcaracterísticas)", "Respondidos por ambos": int(both.sum()),
                      "Acuerdo %": round(100 * exact.mean(), 1) if both.any() else None})
        pending = int((~both).sum())
        diff_mask = ((a[SUBCATS + ["not_security"]] != b[SUBCATS + ["not_security"]]).any(axis=1)) | ~both
        rows = []
        for rid in a.index[diff_mask]:
            first_is_a = rng.random() < 0.5
            ra, rb = (a.loc[rid], b.loc[rid]) if first_is_a else (b.loc[rid], a.loc[rid])
            differ = ", ".join(c for c in SUBCATS if ra[c] != rb[c])
            if ra["not_security"] != rb["not_security"]:
                differ = ", ".join(filter(None, [differ, "¿es de seguridad?"]))
            rows.append({"ID": rid, "text": a.loc[rid, "text"], "es": a.loc[rid, "es"],
                         "answer_a": describe(ra), "answer_b": describe(rb),
                         "differ": differ or "falta responder", "a_is": "EXP1" if first_is_a else "EXP2"})
        frame = pd.DataFrame(rows).set_index("ID") if rows else pd.DataFrame(
            columns=["text", "es", "answer_a", "answer_b", "differ", "a_is"])
        write_sheet(wb, f"Consenso_{sheet.split('_', 1)[1]}", f"Consenso · {heading}", frame)
        print(f"{sheet}: {int(both.sum())} respondidos por ambos, {pending} sin responder, {len(frame)} a discutir")
    agreement = pd.DataFrame(table)
    for j, head in enumerate(agreement.columns, start=1):
        cell = summary.cell(row=4, column=j, value=head)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        summary.column_dimensions[get_column_letter(j)].width = 30 if j <= 2 else 14
    for i, values in enumerate(agreement.itertuples(index=False), start=FIRST):
        for j, value in enumerate(values, start=1):
            cell = summary.cell(row=i, column=j, value=None if pd.isna(value) else value)
            cell.font, cell.border = Font(name=FONT, size=10), BORDER
    agreement.to_csv(out / "agreement.csv", index=False, encoding="utf-8-sig")
    wb.save(out / "Consenso.xlsx")
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
