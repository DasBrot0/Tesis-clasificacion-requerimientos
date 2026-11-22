"""Planillas de validación de subcategorías en español: una hoja por dataset, completa.

Uso: python armar_planilla.py <inputs/training> <carpeta de salida>
"""
import csv
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

TRAIN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Confidentiality", "Integrity", "Authenticity", "Resistance"]
ES = {"Confidentiality": "Confidencialidad", "Integrity": "Integridad",
      "Authenticity": "Autenticidad", "Resistance": "Resistencia"}
DATASETS = [("promise", "PROMISE"), ("secreq", "SecReq"), ("dosspre", "DOSSPRE")]
# Casos que conviene mirar con atención (conflictos y dudas de seguridad); solo aparecen en la clave.
FORCED_PATH = OUT / "casos_forzados.csv"
FORCED = ({r["ID"]: r["Motivo"] for r in csv.DictReader(open(FORCED_PATH, encoding="utf-8-sig"))}
          if FORCED_PATH.exists() else {})
# Solo los requisitos de seguridad: esa etiqueta es la del dataset de origen y no se valida;
# los expertos validan únicamente las subcategorías.
DATA = {ds: [r for r in csv.DictReader(open(TRAIN / f"{ds}.csv", encoding="utf-8-sig")) if r["Security"] == "1"]
        for ds, _ in DATASETS}

# ---------------------------------------------------------------- estilos
HEAD = PatternFill("solid", fgColor="1F4E78")
HFONT = Font(bold=True, color="FFFFFF")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top")
BLUE = PatternFill("solid", fgColor="DDEBF7")
YELLOW = PatternFill("solid", fgColor="FFF2CC")
RED = PatternFill("solid", fgColor="F8CBAD")


def header(ws, cols, widths):
    ws.append(cols)
    for i, w in enumerate(widths, 1):
        c = ws.cell(1, i)
        c.fill, c.font, c.border = HEAD, HFONT, BOX
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 32


# ---------------------------------------------------------------- textos
INSTR = [
    ("Validación de subcategorías de seguridad", True),
    ("", False),
    ("Hay una hoja por dataset (PROMISE, SecReq y DOSSPRE) con todos sus requerimientos de seguridad, en "
     "español. Todos ya están clasificados como de seguridad; indique qué subcaracterísticas de seguridad de "
     "la ISO/IEC 25010:2023 expresa cada uno. Valide un dataset completo antes de pasar al siguiente.", False),
    ("", False),
    ("Cómo marcar", True),
    ("1. Lea el requerimiento y la hoja «Definiciones».", False),
    ("2. Escriba X en cada subcategoría que el requerimiento expresa. Puede marcar varias (es multietiqueta).", False),
    ("3. Si no encaja en ninguna de las seis, marque solo «Ninguna».", False),
    ("4. Si el texto no permite decidir, marque «No se puede decidir» y explique en «Observaciones».", False),
    ("5. Si la redacción en español es confusa o parece mal traducida, marque «Problema de redacción».", False),
    ("6. La columna «Estado» se completa sola: «Pendiente» si la fila está vacía y «Revisar» si combinó "
     "«Ninguna» con una subcategoría.", False),
    ("", False),
    ("Reglas", True),
    ("• Marque lo que el requerimiento exige, no lo que sugiere una palabra suelta (por ejemplo, «seguro» o «contraseña» no bastan).", False),
    ("• Trabaje solo: no consulte las respuestas del otro experto. Los desacuerdos se revisan después, en una reunión de consenso.", False),
    ("• No hay respuestas correctas predefinidas: interesa su criterio experto.", False),
]

DEFS = [
    ("Confidencialidad", "Los datos solo son accesibles para quienes están autorizados.",
     "Restringe el acceso o la lectura de información a usuarios, roles o dispositivos autorizados; exige cifrado, ocultación o protección frente a divulgación; exige privacidad.",
     "Solo pide identificar a alguien sin decir qué puede ver o leer (eso es Autenticidad)."),
    ("Integridad", "Se impide el acceso o la modificación no autorizados de datos o programas.",
     "Exige que los datos no se alteren, corrompan ni manipulen; pide MAC, hash o firma para detectar alteraciones; pide validar la exactitud de los datos; restringe quién puede modificar.",
     "Solo restringe la lectura (eso es Confidencialidad)."),
    ("No repudio", "Se puede demostrar que acciones o eventos ocurrieron, de modo que no puedan negarse después.",
     "Exige pruebas, comprobantes, acuses de recibo, firmas o evidencia con valor probatorio frente a una disputa; menciona que una parte no podrá negar haber actuado.",
     "Solo registra eventos para rastrear responsables, sin fin probatorio frente a otra parte (eso es Responsabilidad)."),
    ("Responsabilidad", "Las acciones de una entidad se pueden rastrear de forma única hasta ella.",
     "Exige registros de auditoría, bitácoras, historial o trazabilidad de quién hizo qué y cuándo; reportes de cambios de usuarios o permisos para auditoría.",
     "El registro es solo funcional (p. ej. un historial de compras para el usuario) o «log out» (cerrar sesión)."),
    ("Autenticidad", "Se puede demostrar que un sujeto o recurso es quien dice ser.",
     "Exige autenticar o identificar usuarios, dispositivos, tarjetas o componentes (contraseña, PIN, certificado, credencial, verificación de identidad u origen).",
     "Habla de permisos después de autenticarse sin pedir verificar la identidad."),
    ("Resistencia", "El sistema sigue operando mientras está bajo ataque de un actor malicioso.",
     "Pide resistir, detectar, bloquear o recuperarse de ataques, abuso intencional, malware, intrusiones, repetición de mensajes o intentos reiterados de acceso.",
     "Pide solo disponibilidad o tolerancia a fallas no maliciosas."),
    ("Ninguna", "Es de seguridad, pero no expresa ninguna de las seis.",
     "Requerimientos genéricos («el sistema debe ser seguro»), de respaldos, disponibilidad o mantenibilidad.",
     "Expresa alguna de las seis: no fuerce «Ninguna»."),
    ("No se puede decidir", "El texto no basta para saber si una subcategoría aplica.",
     "Por ejemplo, «la tarjeta debe verificar los datos» sin decir para qué.",
     "Es distinto de «Ninguna»: aquí falta información en el texto."),
]

CASOS = [
    "Cierre de sesión automático (por inactividad, tras un tiempo o al salir): Resistencia.",
    "Control de sesiones simultáneas (una sola sesión a la vez): Resistencia; añada Autenticidad si además verifica la identidad.",
    "Robustez, longitud o caducidad de contraseñas: Autenticidad y Resistencia.",
    "Corrupción o manipulación de datos causada por personas o sistemas: Integridad y Resistencia.",
    "Bloqueo tras varios intentos fallidos, captcha o palabra aleatoria contra ataques: Resistencia. Añada Autenticidad si habla de verificar credenciales.",
    "Control de acceso por roles: Confidencialidad si limita quién puede ver; Integridad si limita quién puede crear, modificar o borrar. Pueden ir ambas.",
    "Firmas digitales, MAC y hash: Integridad. Añada Autenticidad si verifican el origen o la identidad del emisor, y No repudio solo si se menciona que el firmante no podrá negar su acción o que sirve de prueba.",
    "Registro de transacciones o eventos: Responsabilidad si el fin es rastrear quién hizo qué y cuándo; añada No repudio solo si habla de pruebas, comprobantes, disputas o de impedir que una parte niegue haber actuado.",
    "Reportes de altas de usuarios o cambios de permisos para auditoría: Responsabilidad.",
    "Cifrado y canales seguros: Confidencialidad. Añada Integridad solo si menciona detectar alteraciones.",
    "Autenticación (contraseña, PIN, certificado): Autenticidad. «Autenticarse antes de acceder» no implica Confidencialidad salvo que hable de proteger información.",
    "Hash de contraseñas para almacenarlas: Confidencialidad (evita su divulgación); hash para detectar alteraciones: Integridad.",
    "Detección o prevención de intrusiones, antivirus, firewall, detección de anomalías o de escaneo de puertos: Resistencia.",
    "Tolerancia a fallos, respaldos, recuperación de errores o rendimiento sin actor malicioso: «Ninguna».",
    "Validación de entradas (inyección SQL, XSS, entradas malformadas): Resistencia, e Integridad si menciona proteger los datos.",
    "Disponibilidad, respaldos y mantenibilidad: no son subcaracterísticas de seguridad en la norma; si solo trata de eso, «Ninguna».",
    "Privacidad y datos personales: Confidencialidad.",
    "Dos objetivos en un mismo requerimiento: marque ambos; no elija el principal.",
]

EJEMPLOS = [
    ("El sistema deberá cifrar todas las contraseñas almacenadas.", "Confidencialidad"),
    ("La tarjeta deberá verificar el MAC de cada mensaje recibido y descartarlo si no es válido.", "Integridad, Autenticidad"),
    ("Tras tres intentos fallidos de inicio de sesión, la cuenta deberá bloquearse.", "Resistencia"),
    ("Solo los administradores podrán ver y modificar los registros de usuarios.", "Confidencialidad, Integridad"),
    ("Cada transacción deberá registrarse con el identificador del usuario y la hora.", "Responsabilidad"),
    ("El recibo firmado servirá como prueba en caso de disputa.", "No repudio, Integridad"),
    ("El sistema deberá ser seguro.", "Ninguna (genérico)"),
]

# Las seis subcaracterísticas de seguridad de la ISO/IEC 25010:2023, en el orden de la norma.
COLS = ["N.º", "ID", "Requerimiento", "Confidencialidad", "Integridad", "No repudio", "Responsabilidad",
        "Autenticidad", "Resistencia", "Ninguna", "No se puede decidir", "Problema de redacción", "Observaciones",
        "Estado"]
WIDTHS = [6, 14, 75, 9, 9, 9, 11, 9, 9, 9, 10, 11, 40, 11]


# ---------------------------------------------------------------- hojas
def instructions(wb, expert):
    ws = wb.active
    ws.title = "Instrucciones"
    ws.column_dimensions["A"].width = 120
    for text, bold in INSTR:
        ws.append([text])
        c = ws.cell(ws.max_row, 1)
        c.alignment = WRAP
        if bold:
            c.font = Font(bold=True, size=13 if ws.max_row == 1 else 11)
    ws.append([""])
    ws.append([f"Experto: {expert}"])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    for ds, name in DATASETS:
        ws.append([f"{name}: {len(DATA[ds])} requerimientos de seguridad"])


def definitions(wb):
    wd = wb.create_sheet("Definiciones")
    header(wd, ["Etiqueta", "Definición (ISO/IEC 25010:2023)", "Se marca cuando el requerimiento…",
                "No se marca cuando…"], [20, 45, 70, 50])
    wd.freeze_panes = "A2"
    for row in DEFS:
        wd.append(list(row))
    wd.append([])
    wd.append(["Casos de frontera"])
    wd.cell(wd.max_row, 1).font = Font(bold=True)
    for t in CASOS:
        wd.append(["", t])
        wd.merge_cells(start_row=wd.max_row, start_column=2, end_row=wd.max_row, end_column=4)
    wd.append([])
    wd.append(["Ejemplos"])
    wd.cell(wd.max_row, 1).font = Font(bold=True)
    for req, lab in EJEMPLOS:
        wd.append(["", req, lab])
    for row in wd.iter_rows(min_row=2):
        for c in row:
            c.alignment = WRAP


def dataset_sheet(wb, ds, name):
    ws = wb.create_sheet(name)
    header(ws, COLS, WIDTHS)
    ws.freeze_panes = "D2"
    dv = DataValidation(type="list", formula1='"X"', allow_blank=True,
                        error="Escriba X o deje la celda vacía.", errorTitle="Valor no válido")
    ws.add_data_validation(dv)
    rows = DATA[ds]
    for i, r in enumerate(rows, 1):
        n = i + 1
        # Pendiente: nada marcado; Revisar: «Ninguna» junto a una subcategoría.
        # D:I = las seis subcategorías, J = Ninguna, K = No se puede decidir.
        estado = (f'=IF(COUNTA(D{n}:K{n})=0,"Pendiente",'
                  f'IF(AND(J{n}<>"",COUNTA(D{n}:I{n})>0),"Revisar","OK"))')
        ws.append([i, r["ID"], r["Requirement_ES"]] + [""] * 10 + [estado])
        for col in range(1, 15):
            c = ws.cell(n, col)
            c.border = BOX
            c.alignment = WRAP if col in (3, 13) else CENTER
    last = len(rows) + 1
    dv.add(f"D2:L{last}")
    ws.conditional_formatting.add(f"N2:N{last}", FormulaRule(formula=['N2="Pendiente"'], fill=YELLOW))
    ws.conditional_formatting.add(f"N2:N{last}", FormulaRule(formula=['N2="Revisar"'], fill=RED))
    ws.auto_filter.ref = f"A1:N{last}"


def build(expert):
    wb = Workbook()
    instructions(wb, expert)
    definitions(wb)
    for ds, name in DATASETS:
        dataset_sheet(wb, ds, name)
    wb.active = wb.sheetnames.index(DATASETS[0][1])
    path = OUT / f"validacion_subcategorias_{expert}.xlsx"
    wb.save(path)
    return path


paths = [build("EXP1"), build("EXP2")]

# ---------------------------------------------------------------- clave (solo autores)
wb = Workbook()
wb.remove(wb.active)
for ds, name in DATASETS:
    ws = wb.create_sheet(name)
    header(ws, ["N.º", "ID", "Requerimiento (ES)", "Requirement (EN)", "Etiqueta de origen", "Seguridad"]
           + [ES[l] + " (actual)" for l in LABELS] + ["Origen de la subcategoría", "Traducción", "Atención"],
           [6, 14, 60, 60, 14, 10, 10, 10, 10, 10, 18, 24, 40])
    ws.freeze_panes = "D2"
    for i, r in enumerate(DATA[ds], 1):
        ws.append([i, r["ID"], r["Requirement_ES"], r["Requirement_original"], r["Original_label"],
                   int(r["Security"])] + [int(r[l]) for l in LABELS]
                  + [r["Subcategory_label_status"], r["Translation_source"], FORCED.get(r["ID"], "")])
        for c in ws[ws.max_row]:
            c.alignment = WRAP
    ws.auto_filter.ref = f"A1:M{len(DATA[ds]) + 1}"
key = OUT / "clave_etiquetas_actuales_AUTORES.xlsx"
wb.save(key)

for ds, name in DATASETS:
    print(name, len(DATA[ds]), "requerimientos,", sum(r["Security"] == "1" for r in DATA[ds]), "de seguridad")
print(*paths, key, sep="\n")
