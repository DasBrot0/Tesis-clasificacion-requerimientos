"""Build the expert validation workbooks (one per expert) for the labelling and augmentation plans.

EXP1: pilot (E3), blind sample (E4), review of the LLM pre-annotation (E5), synthetic review (G6).
EXP2: pilot (E3), blind sample (E4), DOSSPRE audit (E6) and the contradictory DOSSPRE duplicates.

Usage: py -I scripts/expert_review/build_validation_sheets.py
Output: inputs/interim/expert_review/Validacion_EXP1.xlsx, Validacion_EXP2.xlsx, manifest.csv
"""
import re
from math import ceil
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "inputs" / "interim" / "expert_review"
SEED = 42
SUBCATS = ["Confidentiality", "Integrity", "Authenticity", "Resistance"]
N_PILOT, N_BLIND, N_DOSSPRE, MIN_POS_BLIND = 30, 75, 100, 10

FONT = "Arial"
NAVY, BLUE_LIGHT = "1F3864", "D9E2F3"
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
REF_FILL = PatternFill("solid", fgColor="F2F2F2")
HEAD_FILL = PatternFill("solid", fgColor=NAVY)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top", wrap_text=True)
FIRST = 5  # first data row; rows 1-3 are title and instructions, row 4 the header


def clean(text):
    """Drop the stray quotes and cp1252 escapes that some original rows carry (e.g. user\\92s)."""
    return str(text).replace("\\92", "'").replace("\x92", "'").strip(" \t\n\"'")


def norm(text):
    """Match key that ignores case, spacing and punctuation (the label and training files differ in those)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", clean(text).lower()).split())


# ---------------------------------------------------------------- data and samples
pre = pd.read_csv(ROOT / "inputs/interim/labels/son_prelabels.csv")
train = pd.concat([pd.read_csv(ROOT / f"inputs/training/{d}.csv") for d in ("promise", "secreq", "dosspre")])
es_by_text = dict(zip(train.Requirement_original.map(norm), train.Requirement_ES))
pre["es"] = pre.text.map(norm).map(es_by_text)
pre["text"] = pre.text.map(clean)
pre["llm"] = pre[SUBCATS].apply(lambda r: ", ".join(c for c in SUBCATS if r[c] == 1) or "Ninguna", axis=1)

pilot = pre.groupby("dataset").sample(frac=N_PILOT / len(pre), random_state=SEED)
rest = pre.drop(pilot.index)
# Blind sample: at least MIN_POS_BLIND LLM-positives per subcategory (rarest first), then random fill.
blind_idx = set()
for c in sorted(SUBCATS, key=lambda c: rest[c].sum()):
    have = sum(rest.loc[list(blind_idx), c]) if blind_idx else 0
    pool = rest[(rest[c] == 1) & ~rest.index.isin(blind_idx)]
    blind_idx |= set(pool.sample(min(len(pool), max(0, MIN_POS_BLIND - have)), random_state=SEED).index)
fill = rest.drop(list(blind_idx)).sample(N_BLIND - len(blind_idx), random_state=SEED)
blind = pd.concat([rest.loc[list(blind_idx)], fill]).sample(frac=1, random_state=SEED)
review = rest.drop(blind.index).sample(frac=1, random_state=SEED)

dos = train[(train.Dataset == "dosspre") & (train.Security == 1)].copy()
dos["Requirement_original"] = dos.Requirement_original.map(clean)
dos["cls"] = dos.Original_label.where(dos.Original_label.map(dos.Original_label.value_counts()) >= 10, "Otras")
quota = (dos.cls.value_counts() * N_DOSSPRE / len(dos)).round().astype(int)
dos_sample = pd.concat([dos[dos.cls == k].sample(n, random_state=SEED) for k, n in quota.items()])
dos_sample = dos_sample.sample(frac=1, random_state=SEED)

syn = pd.concat([pd.read_csv(ROOT / f"inputs/interim/augmentation/synthetic_{d}.csv")
                 for d in ("promise", "secreq", "dosspre")])
syn["intended"] = syn[SUBCATS].apply(lambda r: ", ".join(c for c in SUBCATS if r[c] == 1) or "Ninguna", axis=1)
syn = syn.sample(frac=1, random_state=SEED).sort_values("Dataset", kind="stable", key=lambda s: s.map(
    {"promise": 0, "secreq": 1, "dosspre": 2}))

DUPLICATES = [  # PLAN_ETIQUETADO_SUBCATEGORIAS.md, E6 step 7 (full text from DOSSPRE 3.0)
    ("The system shall be able to open the door when the pin is correct.", "Seguridad: Authorisation", "No seguridad: Performance"),
    ("The system shall be able to do random check on connection.", "No seguridad: Performance", "Seguridad: Availability"),
    ("The system shall allow a user to sign a document on behalf of the group using their secret key.", "Seguridad: Authorisation", "Seguridad: Integrity"),
    ("The system shall allow anomalies to be curbed after detection.", "Seguridad: Intrusion detection", "Seguridad: Immunity"),
    ("The system has clearly defined links and buttons that show the various functionalities that the user may require.", "No seguridad: Functionality", "No seguridad: Usability"),
]


# ---------------------------------------------------------------- sheet helpers
def title(ws, text, subtitle, note):
    ws["A1"] = text
    ws["A1"].font = Font(name=FONT, size=14, bold=True, color=NAVY)
    ws["A2"] = subtitle
    ws["A2"].font = Font(name=FONT, size=10, italic=True, color="404040")
    ws["A3"] = note
    ws["A3"].font = Font(name=FONT, size=10, color="404040")
    ws.sheet_view.showGridLines = False


def table(ws, columns, rows, freeze):
    """columns: list of (header, width, kind) with kind in ref | text | input | yesno | formula | hidden."""
    for j, (head, width, kind) in enumerate(columns, start=1):
        cell = ws.cell(row=4, column=j, value=head)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill, cell.alignment, cell.border = HEAD_FILL, CENTER, BORDER
        ws.column_dimensions[get_column_letter(j)].width = width
        ws.column_dimensions[get_column_letter(j)].hidden = kind == "hidden"
    ws.row_dimensions[4].height = 42
    yesno = DataValidation(type="list", formula1='"Sí,No"', allow_blank=True,
                           error="Elige Sí o No de la lista.", errorTitle="Valor no válido")
    ws.add_data_validation(yesno)
    last = FIRST + len(rows) - 1
    for i, values in enumerate(rows):
        r = FIRST + i
        longest = 1
        for j, ((_, width, kind), value) in enumerate(zip(columns, values), start=1):
            cell = ws.cell(row=r, column=j, value=value)
            cell.font = Font(name=FONT, size=10, italic=kind == "ref", color="595959" if kind == "ref" else "000000")
            cell.border = BORDER
            cell.alignment = CENTER if kind in ("yesno", "formula") else WRAP_TOP
            if kind in ("input", "yesno"):
                cell.fill = INPUT_FILL
            elif kind == "ref":
                cell.fill = REF_FILL
            if kind == "yesno":
                yesno.add(cell)
            if kind in ("text", "ref") and isinstance(value, str):
                longest = max(longest, ceil(len(value) / (width * 1.15)))
        ws.row_dimensions[r].height = max(18, 14 * longest + 4)
    ws.freeze_panes = freeze
    ws.auto_filter.ref = f"A4:{get_column_letter(len(columns))}{last}"
    return last


def color_yes(ws, cols, last):
    green, red = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="F8CBAD")
    for col in cols:
        rng = f"{col}{FIRST}:{col}{last}"
        ws.conditional_formatting.add(rng, FormulaRule(formula=[f'{col}{FIRST}="Sí"'], fill=green))
        ws.conditional_formatting.add(rng, FormulaRule(formula=[f'{col}{FIRST}="No"'], fill=red))


def color_status(ws, col, last):
    rng = f"{col}{FIRST}:{col}{last}"
    for word, colour in (("Completo", "C6EFCE"), ("Revisar", "F8CBAD"), ("Pendiente", "FFFFFF")):
        ws.conditional_formatting.add(rng, FormulaRule(formula=[f'{col}{FIRST}="{word}"'],
                                                        fill=PatternFill("solid", fgColor=colour)))


LABEL_COLS = [(c, 14, "yesno") for c in SUBCATS] + [("Ninguna aplica", 12, "yesno"),
                                                      ("No me parece de seguridad", 13, "yesno")]


def blind_sheet(wb, name, heading, frame, text_col, es_col, id_col, hidden):
    ws = wb.create_sheet(name)
    title(ws, heading, "Anota a ciegas: marca «Sí» solo en las subcaracterísticas que el requisito exige "
                       "(vacío = no aplica). Si es de seguridad pero ninguna aplica, marca «Sí» en «Ninguna aplica».",
          "Juzga el texto en inglés; la traducción es solo de referencia. La columna «Estado» se llena sola.")
    cols = [("N°", 5, "ref"), ("ID", 13, "ref"), ("Requisito (inglés)", 62, "text"),
            ("Traducción al español (referencia)", 44, "ref")] + LABEL_COLS + [
           ("Duda: ¿qué subcaracterística no puedes decidir?", 20, "input"), ("Comentario", 26, "input"),
           ("Estado", 11, "formula")] + [(h, 10, "hidden") for h, _ in hidden]
    rows = []
    for k, (_, row) in enumerate(frame.iterrows(), start=1):
        r = FIRST + k - 1
        status = (f'=IF(AND(I{r}="Sí",COUNTIF(E{r}:H{r},"Sí")>0),"Revisar",'
                  f'IF(OR(COUNTIF(E{r}:H{r},"Sí")>0,I{r}="Sí"),"Completo","Pendiente"))')
        rows.append([k, row[id_col], row[text_col], row[es_col] if pd.notna(row[es_col]) else "",
                     None, None, None, None, None, None, None, None, status] + [row[c] for _, c in hidden])
    last = table(ws, cols, rows, "E5")
    color_yes(ws, "EFGHIJ", last)
    color_status(ws, "M", last)
    ws["M4"].comment = Comment("«Revisar»: marcaste «Ninguna aplica» y también alguna subcaracterística.", "Tesis")
    return name, len(rows), "M"


def review_sheet(wb, frame):
    ws = wb.create_sheet("3_Revision_LLM")
    title(ws, "Hoja 3 · Revisión de la preanotación del LLM (paso E5)",
          "Las columnas amarillas vienen llenas con la propuesta del LLM. Corrige lo que no estés de acuerdo y "
          "marca «Sí» en «Revisado» en cada fila.",
          "Esta hoja se usa solo después de comparar la hoja 2 con el LLM. Si te parece que no hay que revisarla "
          "todavía, déjala para la segunda sesión.")
    cols = [("N°", 5, "ref"), ("ID", 13, "ref"), ("Requisito (inglés)", 58, "text"),
            ("Traducción al español (referencia)", 40, "ref"), ("Propuesta del LLM", 18, "ref"),
            ("Justificación del LLM", 40, "ref")] + [(c, 14, "yesno") for c in SUBCATS] + [
           ("Ninguna aplica", 12, "yesno"), ("No me parece de seguridad", 13, "yesno"),
           ("Revisado", 11, "yesno"), ("Comentario", 26, "input"), ("¿Cambiaste algo?", 11, "formula"),
           ("Estado", 11, "formula")] + [(f"LLM_{c}", 8, "hidden") for c in SUBCATS]
    rows = []
    for k, (_, row) in enumerate(frame.iterrows(), start=1):
        r = FIRST + k - 1
        llm = ["Sí" if row[c] == 1 else None for c in SUBCATS]
        changed = "=IF(" + "+".join(f'(({col}{r}="Sí")<>({h}{r}=1))'
                                     for col, h in zip("GHIJ", "QRST"))+ '>0,"Sí","No")'
        status = (f'=IF(M{r}<>"Sí","Pendiente",IF(AND(K{r}="Sí",COUNTIF(G{r}:J{r},"Sí")>0),"Revisar",'
                  f'IF(OR(COUNTIF(G{r}:J{r},"Sí")>0,K{r}="Sí"),"Completo","Revisar")))')
        rows.append([k, row["id"], row["text"], row["es"] if pd.notna(row["es"]) else "", row["llm"],
                     row["rationale"]] + llm + ["Sí" if row["llm"] == "Ninguna" else None, None, None, None,
                                                changed, status] + [int(row[c]) for c in SUBCATS])
    last = table(ws, cols, rows, "G5")
    color_yes(ws, "GHIJKLM", last)
    color_status(ws, "P", last)
    return "3_Revision_LLM", len(rows), "P"


def synthetic_sheet(wb, frame):
    ws = wb.create_sheet("4_Sinteticos")
    title(ws, "Hoja 4 · Revisión de los requisitos sintéticos (paso G6)",
          "Requisitos generados por un LLM para el entrenamiento. Responde las dos preguntas amarillas; "
          "la decisión se calcula sola.",
          "«Etiqueta correcta» = expresa exactamente esas subcaracterísticas y ninguna otra. "
          "«Bien redactado» = un requisito plausible, claro y verificable.")
    cols = [("N°", 5, "ref"), ("ID", 17, "ref"), ("Dataset", 10, "ref"), ("Requisito sintético (inglés)", 58, "text"),
            ("Traducción al español (referencia)", 40, "ref"), ("Tipo de software", 16, "ref"),
            ("Etiqueta pretendida", 20, "ref"), ("¿La etiqueta es correcta?", 13, "yesno"),
            ("¿Es un requisito plausible y bien redactado?", 15, "yesno"),
            ("Si la etiqueta no es correcta: ¿cuál sería?", 22, "input"), ("Comentario", 24, "input"),
            ("Decisión", 16, "formula")]
    rows = []
    for k, (_, row) in enumerate(frame.iterrows(), start=1):
        r = FIRST + k - 1
        decision = (f'=IF(H{r}="No","Rechazado por etiqueta",IF(I{r}="No","Rechazado por calidad",'
                    f'IF(AND(H{r}="Sí",I{r}="Sí"),"Aceptado","")))')
        rows.append([k, row["ID"], row["Dataset"].upper() if row["Dataset"] != "secreq" else "SecReq",
                     row["Requirement_original"], row["Requirement_ES"], row["Domain"], row["intended"],
                     None, None, None, None, decision])
    last = table(ws, cols, rows, "H5")
    color_yes(ws, "HI", last)
    rng = f"L{FIRST}:L{last}"
    ws.conditional_formatting.add(rng, FormulaRule(formula=[f'L{FIRST}="Aceptado"'],
                                                    fill=PatternFill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(rng, FormulaRule(formula=[f'LEFT(L{FIRST},9)="Rechazado"'],
                                                    fill=PatternFill("solid", fgColor="F8CBAD")))
    return last


def duplicates_sheet(wb):
    ws = wb.create_sheet("4_Duplicados")
    title(ws, "Hoja 4 · Duplicados de DOSSPRE con etiquetas contradictorias (paso E6.7)",
          "El dataset original trae cada requisito dos veces con etiquetas distintas. Decide la etiqueta correcta.",
          "Primero indica si es de seguridad; si lo es, marca sus subcaracterísticas como en las otras hojas.")
    cols = [("N°", 5, "ref"), ("Requisito (inglés)", 58, "text"), ("Etiqueta de la copia 1", 22, "ref"),
            ("Etiqueta de la copia 2", 22, "ref"), ("¿Es de seguridad?", 12, "yesno")] + [
           (c, 14, "yesno") for c in SUBCATS] + [("Ninguna aplica", 12, "yesno"), ("Comentario", 26, "input"),
                                                 ("Estado", 11, "formula")]
    rows = []
    for k, (text, c1, c2) in enumerate(DUPLICATES, start=1):
        r = FIRST + k - 1
        status = (f'=IF(E{r}="No","Completo",IF(AND(E{r}="Sí",OR(COUNTIF(F{r}:I{r},"Sí")>0,J{r}="Sí")),'
                  f'"Completo","Pendiente"))')
        rows.append([k, text, c1, c2, None, None, None, None, None, None, None, status])
    last = table(ws, cols, rows, "E5")
    color_yes(ws, "EFGHIJ", last)
    color_status(ws, "L", last)
    return "4_Duplicados", len(rows), "L"


DEFINITIONS = [
    ("Confidentiality", "Los datos solo son accesibles para quienes están autorizados.",
     "Restringe quién puede ver o leer información; exige cifrado, ocultación o privacidad.",
     "Solo pide identificar a alguien, sin decir qué puede ver (eso es Authenticity)."),
    ("Integrity", "Se impide el acceso o la modificación no autorizados de datos o programas.",
     "Exige que los datos no se alteren ni corrompan; MAC, hash o firma para detectar cambios; "
     "restringe quién puede modificar.", "Solo restringe la lectura (Confidentiality)."),
    ("Authenticity", "Se puede demostrar que un sujeto o recurso es quien dice ser.",
     "Exige autenticar o identificar usuarios, dispositivos o componentes (contraseña, PIN, certificado).",
     "Habla de permisos después de autenticarse, sin verificar identidad."),
    ("Resistance", "El sistema sigue operando mientras está bajo ataque de un actor malicioso.",
     "Pide resistir, detectar, bloquear o recuperarse de ataques, intrusiones, malware o intentos reiterados.",
     "Solo pide disponibilidad o tolerancia a fallas no maliciosas (fuera de alcance)."),
]
RULES = [  # draft v1 of the annotation guide (contexto_sesion/guia_anotacion_subcategorias_v1_provisional.md)
    ("Bloqueo tras varios intentos fallidos", "Resistance. Añade Authenticity solo si habla de verificar credenciales."),
    ("Control de acceso por roles o permisos", "Confidentiality si limita quién ve; Integrity si limita quién modifica. Pueden ir ambas."),
    ("Firmas digitales, MAC y hash", "Integrity. Añade Authenticity si verifican el origen o la identidad del emisor."),
    ("Cifrado y canales seguros", "Confidentiality. Añade Integrity solo si menciona detectar alteraciones."),
    ("Expiración de sesión o cierre por inactividad", "Resistance si busca limitar el abuso de un atacante."),
    ("Validación de entradas (inyección SQL, XSS)", "Resistance; añade Integrity si protege los datos de modificación."),
    ("Auditoría, bitácoras y no repudio", "Quedan fuera de alcance en esta tesis: si es lo único que pide, «Ninguna aplica»."),
    ("Disponibilidad, respaldos y mantenibilidad", "No son seguridad en ISO/IEC 25010: «Ninguna aplica» o «No me parece de seguridad»."),
    ("Palabras gatillo («secure», «protected», «password», «log»)", "No bastan para marcar nada. Se marca lo que el requisito exige."),
    ("Requisito con dos objetivos", "Se marcan los dos. No se elige «la principal»."),
]
N_NEW_RULES = 4


def rules_sheet(wb):
    ws = wb.create_sheet("0_Reglas")
    title(ws, "Hoja 0 · Reglas para casos de frontera (antes del piloto)",
          "Son un borrador de los autores para decidir los casos ambiguos. Indica si estás de acuerdo con cada una "
          "y, si no, cómo la redactarías.",
          "Si en el piloto aparece un caso que ninguna regla cubre, agrégalo en las filas vacías del final.")
    cols = [("N°", 5, "ref"), ("Caso", 30, "text"), ("Regla propuesta", 52, "text"),
            ("¿Estás de acuerdo?", 12, "yesno"), ("Si no: ¿cómo la redactarías?", 44, "input"),
            ("Comentario", 30, "input"), ("Estado", 11, "formula")]
    rows = []
    for k, (case, rule) in enumerate(RULES + [("", "")] * N_NEW_RULES, start=1):
        r = FIRST + k - 1
        if case:
            status = (f'=IF(D{r}="Sí","Completo",IF(AND(D{r}="No",E{r}<>""),"Completo",'
                      f'IF(D{r}="No","Revisar","Pendiente")))')
        else:
            status = f'=IF(B{r}="","",IF(C{r}="","Revisar","Completo"))'
        rows.append([k, case or None, rule or None, None, None, None, status])
    last = table(ws, cols, rows, "D5")
    for r in range(last - N_NEW_RULES + 1, last + 1):  # empty rows: the expert writes the case and the rule
        for col in "BC":
            ws[f"{col}{r}"].fill = INPUT_FILL
        ws.row_dimensions[r].height = 30
    ws[f"A{last + 2}"] = "Las filas amarillas del final son para reglas nuevas: escribe el caso y la regla."
    ws[f"A{last + 2}"].font = Font(name=FONT, size=10, italic=True, color="404040")
    color_yes(ws, "D", last)
    color_status(ws, "G", last)
    ws["G4"].comment = Comment("«Revisar»: marcaste «No» sin proponer otra redacción, o escribiste un caso "
                               "nuevo sin su regla.", "Tesis")
    return "0_Reglas", len(RULES), "G"


def instructions(wb, sheets, minutes):
    ws = wb.active
    ws.title = "Instrucciones"
    ws.sheet_view.showGridLines = False
    for col, width in zip("ABCD", (22, 46, 52, 46)):
        ws.column_dimensions[col].width = width
    r = 1

    def put(text, size=10, bold=False, color="000000", fill=None, merge=True, height=None):
        nonlocal r
        ws.cell(row=r, column=1, value=text).font = Font(name=FONT, size=size, bold=bold, color=color)
        ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        if fill:
            ws.cell(row=r, column=1).fill = fill
        if merge:
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
        if height:
            ws.row_dimensions[r].height = height
        r += 1

    put("Validación de etiquetas de requisitos de seguridad", 16, True, NAVY, height=26)
    put("Tesis: Comparación de modelos de deep learning para la clasificación multietiqueta de requisitos de "
        "seguridad en español (Universidad de Lima).", 10, color="404040", height=28)
    r += 1
    put("¿Qué te pedimos?", 12, True, NAVY)
    put("Indicar qué subcaracterísticas de seguridad de ISO/IEC 25010:2023 expresa cada requisito. Es una tarea "
        "multietiqueta: un requisito puede expresar varias o ninguna. Tus respuestas son la referencia contra la "
        f"que se evalúan los modelos y el LLM que preetiquetó los datos. Tiempo estimado: {minutes}.", height=58)
    put("Anota de forma independiente: no consultes a otros expertos ni la preanotación salvo en la hoja que la "
        "muestra. Si dudas, usa la columna «Duda» en lugar de adivinar.", height=30)
    r += 1
    put("Hojas, en este orden", 12, True, NAVY)
    for name, what in sheets:
        ws.cell(row=r, column=1, value=name).font = Font(name=FONT, bold=True, size=10)
        ws.cell(row=r, column=1).alignment = Alignment(vertical="top")
        ws.cell(row=r, column=2, value=what).font = Font(name=FONT, size=10)
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 30
        r += 1
    r += 1
    put("Leyenda", 12, True, NAVY)
    for fill, text in ((INPUT_FILL, "Celda para llenar. Las de Sí/No tienen una lista desplegable."),
                       (REF_FILL, "Referencia: no se edita."),
                       (PatternFill("solid", fgColor="C6EFCE"), "Estado «Completo» o respuesta «Sí»."),
                       (PatternFill("solid", fgColor="F8CBAD"), "Estado «Revisar» (respuesta incoherente) o respuesta «No».")):
        ws.cell(row=r, column=1).fill = fill
        ws.cell(row=r, column=1).border = BORDER
        ws.cell(row=r, column=2, value=text).font = Font(name=FONT, size=10)
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        r += 1
    r += 1
    put("Definiciones (ISO/IEC 25010:2023)", 12, True, NAVY)
    for j, head in enumerate(("Subcaracterística", "Definición", "Se marca cuando el requisito…", "No se marca cuando…"), 1):
        c = ws.cell(row=r, column=j, value=head)
        c.font, c.fill, c.alignment, c.border = Font(name=FONT, bold=True, color="FFFFFF", size=10), HEAD_FILL, CENTER, BORDER
    r += 1
    for values in DEFINITIONS:
        for j, value in enumerate(values, 1):
            c = ws.cell(row=r, column=j, value=value)
            c.font, c.alignment, c.border = Font(name=FONT, size=10, bold=j == 1), WRAP_TOP, BORDER
        ws.row_dimensions[r].height = 44
        r += 1
    r += 1
    put("Reglas para casos de frontera", 12, True, NAVY)
    put("Están en la hoja 0_Reglas. Revísalas antes del piloto: ahí indicas si estás de acuerdo con cada una y "
        "cómo la cambiarías. Aplícalas (con tus correcciones) en todas las hojas.", height=30)
    r += 1
    put("Ejemplo de una fila llena (hojas 1 y 2)", 12, True, NAVY)
    example = [("Requisito", "After three failed login attempts the account shall be locked."),
               ("Confidentiality / Integrity", "(vacío) / (vacío)"), ("Authenticity", "(vacío): no pide verificar credenciales"),
               ("Resistance", "Sí: contiene un ataque de fuerza bruta (regla 1)"), ("Ninguna aplica", "(vacío)"),
               ("Estado", "Completo (se llena solo)")]
    for k, v in example:
        ws.cell(row=r, column=1, value=k).font = Font(name=FONT, bold=True, size=10)
        ws.cell(row=r, column=2, value=v).font = Font(name=FONT, size=10)
        ws.cell(row=r, column=2).fill = INPUT_FILL
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        r += 1
    r += 1
    put("Al terminar, guarda el archivo con tu nombre al final (por ejemplo, Validacion_EXP1_Perez.xlsx) y "
        "envíaselo a los autores. Gracias por tu tiempo.", 10, True, height=30)


def summary(wb, progress, synthetic_last=None):
    ws = wb.create_sheet("Resumen", 1)
    title(ws, "Resumen de avance", "Se actualiza solo a medida que llenas las hojas.", "")
    for col, width in zip("ABCDEFGH", (26, 12, 12, 12, 14, 14, 14, 14)):
        ws.column_dimensions[col].width = width
    heads = ("Hoja", "Ítems", "Completos", "Avance")
    for j, h in enumerate(heads, 1):
        c = ws.cell(row=4, column=j, value=h)
        c.font, c.fill, c.alignment, c.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD_FILL, CENTER, BORDER
    r = 5
    for name, n, col, done in progress:
        rng = f"'{name}'!{col}{FIRST}:{col}{FIRST + n - 1}"
        values = [name, f"=COUNTA('{name}'!B{FIRST}:B{FIRST + n - 1})", f'=COUNTIF({rng},"{done}")' if done != "*" else f'=COUNTIF({rng},"?*")',
                  f"=IF(B{r}=0,0,C{r}/B{r})"]
        for j, v in enumerate(values, 1):
            c = ws.cell(row=r, column=j, value=v)
            c.font, c.border = Font(name=FONT, size=10), BORDER
            c.alignment = Alignment(horizontal="left" if j == 1 else "center")
        ws.cell(row=r, column=4).number_format = "0%"
        r += 1
    if synthetic_last is None:
        return
    r += 1
    ws.cell(row=r, column=1, value="Sintéticos por dataset (criterio de Bishop et al., 2026)").font = Font(
        name=FONT, size=12, bold=True, color=NAVY)
    r += 1
    ws.cell(row=r, column=1, value="Nivel de confianza (z)").font = Font(name=FONT, size=10)
    ws.cell(row=r, column=2, value=1.96).font = Font(name=FONT, size=10, color="0000FF")
    ws.cell(row=r, column=2).comment = Comment("z = 1,96 para un intervalo de Wilson al 95 %.", "Tesis")
    ws.cell(row=r, column=3, value="Umbral").font = Font(name=FONT, size=10)
    ws.cell(row=r, column=4, value=0.10).font = Font(name=FONT, size=10, color="0000FF")
    ws.cell(row=r, column=4).number_format = "0%"
    ws.cell(row=r, column=4).comment = Comment("Se acepta el dataset si el límite superior de Wilson de la "
                                               "proporción de rechazados es menor que este umbral "
                                               "(PLAN_AUMENTO_DATOS.md, G6; Bishop et al., 2026).", "Tesis")
    z, thr = f"$B${r}", f"$D${r}"
    r += 1
    heads = ("Dataset", "Revisados", "Aceptados", "Rechazados", "Proporción rechazada",
             "Wilson inferior", "Wilson superior", "¿Cumple?")
    for j, h in enumerate(heads, 1):
        c = ws.cell(row=r, column=j, value=h)
        c.font, c.fill, c.alignment, c.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD_FILL, CENTER, BORDER
    r += 1
    dec = f"'4_Sinteticos'!$L${FIRST}:$L${synthetic_last}"
    ds = f"'4_Sinteticos'!$C${FIRST}:$C${synthetic_last}"
    for name in ("PROMISE", "SecReq", "DOSSPRE"):
        n, p = f"B{r}", f"E{r}"
        wilson = (f"({p}+{z}^2/(2*{n}){{s}}{z}*SQRT({p}*(1-{p})/{n}+{z}^2/(4*{n}^2)))/(1+{z}^2/{n})")
        values = [name, f'=COUNTIFS({ds},A{r},{dec},"?*")', f'=COUNTIFS({ds},A{r},{dec},"Aceptado")',
                  f"=B{r}-C{r}", f"=IF(B{r}=0,0,D{r}/B{r})",
                  f'=IF({n}=0,"",{wilson.format(s="-")})', f'=IF({n}=0,"",{wilson.format(s="+")})',
                  f'=IF(B{r}=0,"",IF(G{r}<{thr},"Sí","No"))']
        for j, v in enumerate(values, 1):
            c = ws.cell(row=r, column=j, value=v)
            c.font, c.border = Font(name=FONT, size=10), BORDER
            c.alignment = Alignment(horizontal="left" if j == 1 else "center")
        for col in "EFG":
            ws[f"{col}{r}"].number_format = "0.0%"
        r += 1


# ---------------------------------------------------------------- workbooks
OUT.mkdir(parents=True, exist_ok=True)
hidden_real = [("Dataset", "dataset")]

wb = Workbook()
instructions(wb, [
    ("0_Reglas", f"{len(RULES)} reglas para los casos ambiguos: ¿estás de acuerdo? Antes del piloto."),
    ("1_Piloto", f"{len(pilot)} requisitos para calibrar la guía. Después se comentan las diferencias con el otro experto."),
    ("2_Muestra_ciegas", f"{len(blind)} requisitos a ciegas. Con ellos se mide el acuerdo entre expertos y con el LLM."),
    ("3_Revision_LLM", f"{len(review)} requisitos con la propuesta del LLM visible: aceptas o corriges. Segunda sesión."),
    ("4_Sinteticos", f"{len(syn)} requisitos generados por LLM: ¿etiqueta correcta? ¿bien redactado?"),
], "unas 4 horas en total, idealmente en dos o tres sesiones")
progress = [rules_sheet(wb) + ("Completo",),
            blind_sheet(wb, "1_Piloto", "Hoja 1 · Piloto (paso E3)", pilot, "text", "es", "id", hidden_real) + ("Completo",),
            blind_sheet(wb, "2_Muestra_ciegas", "Hoja 2 · Muestra a ciegas (paso E4)", blind, "text", "es", "id", hidden_real) + ("Completo",),
            review_sheet(wb, review) + ("Completo",)]
syn_last = synthetic_sheet(wb, syn)
progress.append(("4_Sinteticos", len(syn), "L", "*"))
summary(wb, progress, syn_last)
wb.save(OUT / "Validacion_EXP1.xlsx")

wb = Workbook()
instructions(wb, [
    ("0_Reglas", f"{len(RULES)} reglas para los casos ambiguos: ¿estás de acuerdo? Antes del piloto."),
    ("1_Piloto", f"{len(pilot)} requisitos para calibrar la guía (los mismos que el otro experto)."),
    ("2_Muestra_ciegas", f"{len(blind)} requisitos a ciegas (los mismos que el otro experto)."),
    ("3_DOSSPRE", f"{len(dos_sample)} requisitos de DOSSPRE a ciegas, para auditar cómo se convirtieron sus clases."),
    ("4_Duplicados", f"{len(DUPLICATES)} requisitos que DOSSPRE etiqueta de dos formas distintas."),
], "unas 3 horas en total, idealmente en dos sesiones")
progress = [rules_sheet(wb) + ("Completo",),
            blind_sheet(wb, "1_Piloto", "Hoja 1 · Piloto (paso E3)", pilot, "text", "es", "id", hidden_real) + ("Completo",),
            blind_sheet(wb, "2_Muestra_ciegas", "Hoja 2 · Muestra a ciegas (paso E4)", blind, "text", "es", "id", hidden_real) + ("Completo",),
            blind_sheet(wb, "3_DOSSPRE", "Hoja 3 · Auditoría de DOSSPRE (paso E6)", dos_sample, "Requirement_original",
                        "Requirement_ES", "ID", [("Clase_original", "Original_label")]) + ("Completo",),
            duplicates_sheet(wb) + ("Completo",)]
summary(wb, progress)
wb.save(OUT / "Validacion_EXP2.xlsx")

manifest = pd.concat([
    pd.DataFrame({"sheet": "1_Piloto", "id": pilot["id"], "dataset": pilot.dataset}),
    pd.DataFrame({"sheet": "2_Muestra_ciegas", "id": blind["id"], "dataset": blind.dataset}),
    pd.DataFrame({"sheet": "3_Revision_LLM", "id": review["id"], "dataset": review.dataset}),
    pd.DataFrame({"sheet": "3_DOSSPRE", "id": dos_sample.ID, "dataset": "dosspre"}),
    pd.DataFrame({"sheet": "4_Sinteticos", "id": syn.ID, "dataset": syn.Dataset}),
])
manifest.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
print(f"pilot={len(pilot)} blind={len(blind)} review={len(review)} dosspre={len(dos_sample)} synthetic={len(syn)}")
print("blind LLM positives:", blind[SUBCATS].sum().to_dict())
print("translations found:", pre.es.notna().sum(), "/", len(pre))
print("dosspre quota:", quota.to_dict())
