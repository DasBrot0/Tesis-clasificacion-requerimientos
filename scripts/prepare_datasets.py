#!/usr/bin/env python3
"""Construye los datasets de entrenamiento a partir de los archivos fuente.

    python scripts/prepare_datasets.py

Por cada dataset (PROMISE_exp, SecReq, DOSSPRE) escribe en inputs/training/:

    <dataset>.csv                 tabla completa: Security + cuatro subcategorías
    <dataset>_binary.csv          tarea 1: todos los requisitos, etiqueta Security
    <dataset>_subcategories.csv   tarea 2: solo los requisitos de seguridad, cuatro subcategorías

Los dos archivos por tarea siguen el formato de las fuentes de DOSSPRE ("2.0Binary" con SR/NSR y
"3.0Multi-class" con la clase de seguridad) y tienen los mismos ID y textos que la tabla completa.

Pasos de cada dataset:
  1. Leer la fuente y su texto en español:
       PROMISE_exp  español del antecedente (Limaylla-Lunarejo et al.), sin las filas sintéticas de ChatGPT;
       SecReq       inglés original + traducción DeepL con glosario;
       DOSSPRE      inglés original + traducción DeepL con glosario; Security sale de "2.0Binary".
  2. Etiqueta Security: SE / sec / SR -> 1, el resto -> 0. Availability y Maintainability de
     DOSSPRE pasan a 0 porque en ISO/IEC 25010 no son Seguridad.
  3. Quitar duplicados, primero exactos (y asignar los ID), luego casi duplicados, cada vez sobre
     el inglés y después sobre el español. Un duplicado con etiquetas distintas es un error en
     PROMISE y SecReq; en DOSSPRE se fusiona con la unión de sus etiquetas.
  4. Subcategorías: PROMISE y SecReq toman el voto de 3 corridas de un LLM sobre el inglés
     (inputs/interim/labels/son_prelabels.csv); DOSSPRE las mapea desde su clase (DOSSPRE_CLASS_MAP).
  5. Casi duplicados que se fusionan a mano (inputs/interim/duplicados_manuales.csv).
  6. Limpieza final: comillas de exportación, correcciones revisadas de traducción y de etiquetas.
  7. Guardar la tabla completa y los dos archivos por tarea.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    DATASETS, INTERIM_DIR, LABEL_COLS, ORIGINAL_DIR, SUBCATEGORY_COLS, TEXT_COL, full_table_path, task_file_path,
)

# --------------------------------------------------------------------------- archivos

PROMISE_FILES = ["Balanced_promise_exp_nfr_train.csv", "Spanish_promise_exp_nfr_test_recovered.csv"]
DOSSPRE_ORIGINAL = "DOSSPRE 1.0Original.csv"
DOSSPRE_BINARY = "DOSSPRE 2.0Binary.csv"
SOURCE_EN = INTERIM_DIR / "glossary" / "source_en.csv"                       # inglés de SecReq
DEEPL_TRANSLATION = INTERIM_DIR / "translations" / "deepl_provisional.jsonl"  # inglés -> español
LLM_LABELS = INTERIM_DIR / "labels" / "son_prelabels.csv"                    # subcategorías de PROMISE y SecReq
MANUAL_DUPLICATES = INTERIM_DIR / "duplicados_manuales.csv"                  # ID_eliminado, ID_conservado
TRANSLATION_FIXES = INTERIM_DIR / "translations" / "correcciones_revision.csv"  # ID, ES_anterior, ES_corregido
LABEL_FIXES = INTERIM_DIR / "labels" / "correcciones_etiquetas.csv"          # ID, Antes, Despues

TRANSLATION_SOURCE_DEEPL = "deepl+glossary_provisional"
TRANSLATION_SOURCE_PROMISE = "promise_exp_translated"

COMMON_COLS = ["ID", "Dataset", "Source_file", "Project", "Source_ID", "Requirement_original", TEXT_COL,
               "Translation_source", "Original_label"]
OUTPUT_COLS = [*COMMON_COLS, *LABEL_COLS, "Subcategory_label_status", "Prelabel_evidence"]

# --------------------------------------------------------------------------- etiquetas de origen

# Etiquetas de seguridad de PROMISE y SecReq; cualquier otra categoría de RNF de PROMISE es Security=0.
SECURITY_LABELS = {"se", "sec", "security", "seguridad"}
NON_SECURITY_LABELS = {"nonsec", "nonsecurity", "noseguridad", "a", "po", "l", "ft", "sc", "mn", "lf", "pe", "o", "us"}

# Clase de DOSSPRE (en minúsculas) -> subcategoría ISO/IEC 25010 (None: seguridad sin subcategoría).
DOSSPRE_CLASS_MAP = {
    "confidentiality": "Confidentiality",
    "authorisation": "Confidentiality",
    "integrity": "Integrity",
    "auditing": None,  # Accountability se descartó el 25-10-2026 por falta de datos
    "authentication": "Authenticity",
    "identification": "Authenticity",
    "immunity": "Resistance",
    "intrusiond": "Resistance",
    "detection": "Resistance",
    "survivability": "Resistance",
}
# SR en DOSSPRE (Kadebu et al., 2023), pero Fiabilidad y Mantenibilidad en ISO/IEC 25010: Security=0.
DOSSPRE_OUT_OF_SECURITY = {"availability", "maintainability", "maintenability", "maintanability"}
DOSSPRE_NON_SECURITY = {"functionality", "performance", "usability", "portability"}

# Abreviaturas de las subcategorías en LABEL_FIXES.
LABEL_ABBREVIATIONS = {"CON": "Confidentiality", "INT": "Integrity", "AUT": "Authenticity", "RES": "Resistance"}

# =========================================================================== normalización de texto


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def norm_name(value) -> str:
    """Nombre de columna o etiqueta comparable: minúsculas, sin tildes ni símbolos."""
    return re.sub(r"[^a-z0-9]+", "", strip_accents(str(value)).lower())


def norm_text(value) -> str:
    """Clave de duplicados exactos: minúsculas, sin tildes y con los espacios normalizados."""
    return re.sub(r"\s+", " ", strip_accents(str(value).lower())).strip()


ENGLISH_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def near_duplicate_key(value, drop_articles: bool = False) -> str:
    """Clave de casi duplicados: norm_text sin puntuación y, en inglés, sin artículos."""
    text = re.sub(r"[^\w\s]", " ", norm_text(value))
    if drop_articles:
        text = ENGLISH_ARTICLES.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).replace("﻿", " ")).strip()
    return text or None


SECREQ_QUOTED_EN = re.compile(r'^"\s*""(.*)"""$', re.S)
SECREQ_QUOTED_ES = re.compile(r'^(«\s*»\s*«|"\s*"")\s*')


def strip_export_quotes(value):
    """Quita las comillas de exportación que envuelven el requisito entero.

    SecReq trae 37 requisitos entre comillas CSV escapadas dos veces, que DeepL convirtió en «»«texto»»»;
    en DOSSPRE el inglés acaba en un apóstrofo suelto que DeepL tradujo como «texto». Las comillas
    internas (p. ej. «OPEN») no se tocan.
    """
    if not isinstance(value, str):
        return value
    text = value
    if match := SECREQ_QUOTED_EN.match(text):
        text = match.group(1)
    elif SECREQ_QUOTED_ES.match(text):
        text = SECREQ_QUOTED_ES.sub("", text)
        text = re.sub(r'[«»"\s,]+$', "", text)
        text += "»" * (text.count("«") - text.count("»"))
    elif text.count("'") == 1 and re.search(r"[\w).,]'[.,]?$", text):
        text = re.sub(r"'([.,]?)$", r"\1", text)
    elif text.count("«") == 1 and text.count("»") == 1 and re.fullmatch(r"«[^«»]*»[.,]?", text):
        text = re.sub(r"^«(.*)»([.,]?)$", r"\1\2", text)
    return text.rstrip(" ,") if text != value else text


def security_label(value) -> int | None:
    """1 seguridad, 0 no seguridad, None si la etiqueta no es reconocible."""
    if value is None or pd.isna(value):
        return None
    name = norm_name(value)
    if name in SECURITY_LABELS or name in {"1", "true"}:
        return 1
    if name in NON_SECURITY_LABELS or name in {"0", "false"}:
        return 0
    return None


def deepl_translations() -> dict[str, str]:
    """Inglés normalizado -> traducción DeepL con glosario."""
    with open(DEEPL_TRANSLATION, encoding="utf-8") as fh:
        return {norm_text(row["en"]): row["es"] for row in map(json.loads, fh)}


# =========================================================================== paso 1-2: leer las fuentes


def read_csv_any_format(path: Path) -> pd.DataFrame:
    """Lee un CSV probando las codificaciones y separadores con que vienen las fuentes."""
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
            except Exception:
                continue
            if df.shape[1] >= 2:
                return df
    raise ValueError(f"No se pudo leer {path}.")


def find_column(df: pd.DataFrame, *aliases: str) -> str | None:
    columns = {norm_name(c): c for c in df.columns}
    return next((columns[norm_name(a)] for a in aliases if norm_name(a) in columns), None)


def read_promise() -> pd.DataFrame:
    """PROMISE_exp en español, sin las filas generadas con ChatGPT del dataset balanceado publicado
    (ID o PROJECT originales vacíos, o PROYECTO = CHATGPT)."""
    rows = []
    for name in PROMISE_FILES:
        df = read_csv_any_format(ORIGINAL_DIR / name)
        english = find_column(df, "REQUIREMENT", "Requirement_original", "Requirement")
        spanish = find_column(df, "REQUISITO", "Requirement_ES", "Requisito", "Spanish_requirement")
        label = find_column(df, "TIPO", "LABEL", "Original_label", "Label")
        project = find_column(df, "PROJECT", "Project")
        translated_project = find_column(df, "PROYECTO", "Proyecto")
        source_id = find_column(df, "ID", "Source_ID")
        for idx, row in df.iterrows():
            text, security = clean_text(row[spanish]), security_label(row[label])
            if not text or security is None:
                continue
            is_synthetic = ((source_id and pd.isna(row[source_id])) or (project and pd.isna(row[project]))
                            or (translated_project and norm_name(clean_text(row[translated_project]) or "")
                                in {"chatgpt", "synthetic", "sintetico", "sintetica"}))
            if is_synthetic:
                continue
            rows.append({
                "Source_file": name,
                "Project": clean_text(row[project]) if project else None,
                "Source_ID": clean_text(row[source_id]) if source_id else str(idx),
                "Requirement_original": clean_text(row[english]) if english else None,
                TEXT_COL: text,
                "Original_label": clean_text(row[label]),
                "Security": security,
                "Translation_source": TRANSLATION_SOURCE_PROMISE,
            })
    return pd.DataFrame(rows)


def read_secreq() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_EN, encoding="utf-8-sig")
    source = source[source["Dataset"] == "secreq"]
    spanish = deepl_translations()
    return pd.DataFrame([{
        "Source_file": f"SecReq/{r.Spec}",
        "Project": r.Spec,
        "Source_ID": r.Source_ID,
        "Requirement_original": clean_text(r.Requirement_en),
        TEXT_COL: clean_text(spanish[norm_text(r.Requirement_en)]),
        "Original_label": r.Label,
        "Security": security_label(r.Label),
        "Translation_source": TRANSLATION_SOURCE_DEEPL,
    } for r in source.itertuples()])


def dosspre_english(value) -> str:
    """Los requisitos vienen entre comillas simples y con comillas tipográficas de cp1252."""
    text = str(value).strip().strip("'").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"')).strip()


def read_dosspre() -> pd.DataFrame:
    """DOSSPRE con Security de "2.0Binary" y la subcategoría mapeada desde la clase de "1.0Original"."""
    original = pd.read_csv(ORIGINAL_DIR / DOSSPRE_ORIGINAL, encoding="cp1252")
    binary = pd.read_csv(ORIGINAL_DIR / DOSSPRE_BINARY, encoding="cp1252")
    if len(original) != len(binary) or not original["Project ID"].equals(binary["Project ID"]):
        raise ValueError("DOSSPRE 1.0 y 2.0 no están alineados fila a fila.")
    out = pd.DataFrame({
        "Source_file": DOSSPRE_ORIGINAL,
        "Project": original["Project ID"].astype(str),
        "Source_ID": (original.index + 1).astype(str),
        "Requirement_original": original["Requirement"].map(dosspre_english),
        "Original_label": original["Class"].str.strip(),
        "Security": binary["Class"].str.strip().map({"SR": 1, "NSR": 0}),
    })
    if out["Security"].isna().any():
        raise ValueError("DOSSPRE 2.0Binary contiene clases distintas de SR/NSR.")
    classes = out["Original_label"].str.lower()
    unknown = set(classes) - set(DOSSPRE_CLASS_MAP) - DOSSPRE_NON_SECURITY - DOSSPRE_OUT_OF_SECURITY
    if unknown:
        raise ValueError(f"Clases DOSSPRE sin mapear: {sorted(unknown)}")

    # paso 2: Availability y Maintainability salen de Seguridad
    out[SUBCATEGORY_COLS] = 0
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "structural_zero"
    out_of_security = classes.isin(DOSSPRE_OUT_OF_SECURITY) & out["Security"].eq(1)
    out.loc[out_of_security, "Security"] = 0
    out.loc[out_of_security, "Prelabel_evidence"] = "Security=0: SR en DOSSPRE, fuera de Seguridad en ISO/IEC 25010"

    # paso 4 (en DOSSPRE va antes de los duplicados, que se fusionan con la unión de etiquetas)
    for idx in out.index[out["Security"].eq(1)]:
        target = DOSSPRE_CLASS_MAP.get(classes[idx])
        if target:
            out.at[idx, target] = 1
            out.at[idx, "Prelabel_evidence"] = f"{target}: clase DOSSPRE '{out.at[idx, 'Original_label']}'"
        out.at[idx, "Subcategory_label_status"] = "class_mapped" if target else "subcategory_dropped"

    spanish = deepl_translations()
    keys = out["Requirement_original"].map(norm_text)
    missing = sorted(set(keys) - set(spanish))
    if missing:
        raise ValueError(f"dosspre: {len(missing)} textos sin traducción DeepL, p. ej. {missing[:3]}")
    out[TEXT_COL] = keys.map(spanish)
    out["Translation_source"] = TRANSLATION_SOURCE_DEEPL
    return out


# =========================================================================== paso 3: duplicados


def drop_duplicates(out: pd.DataFrame, name: str, merge_conflicts: bool, spanish: bool = False,
                    near: bool = False) -> pd.DataFrame:
    """Deja la primera copia de cada texto repetido (exacto, o casi duplicado con near=True).

    spanish=False compara el inglés original; spanish=True, la traducción (dos inglés distintos
    pueden dar el mismo español). Si las copias tienen etiquetas distintas: error, salvo con
    merge_conflicts, en cuyo caso se conserva la unión de sus etiquetas (DOSSPRE repite requisitos
    entre proyectos con clases distintas, un caso multietiqueta legítimo).
    """
    text = out[TEXT_COL] if spanish else out["Requirement_original"].fillna(out[TEXT_COL])
    out["_key"] = text.map(lambda v: near_duplicate_key(v, drop_articles=not spanish)) if near else text.map(norm_text)
    labels = [col for col in LABEL_COLS if col in out.columns]
    conflicts = out.groupby("_key")[labels].nunique().max(axis=1)
    conflict_keys = set(conflicts[conflicts > 1].index)
    if conflict_keys and not merge_conflicts:
        examples = out[out["_key"].isin(conflict_keys)][[TEXT_COL, "Original_label", *labels]]
        raise ValueError(f"{name}: textos duplicados con etiquetas contradictorias:\n"
                         + examples.head(20).to_string(index=False))
    if conflict_keys:
        groups = out.groupby("_key", sort=False)
        out[labels] = groups[labels].transform("max")
        for col in ("Original_label", "Prelabel_evidence"):
            if col in out.columns:  # sin repetir valores ya fusionados en una pasada anterior
                out[col] = groups[col].transform(
                    lambda part: " | ".join(dict.fromkeys(p for v in part if v for p in str(v).split(" | "))))
        out.loc[out["_key"].isin(conflict_keys), "Subcategory_label_status"] = "class_mapped_merged_duplicates"
        print(f"{name}: {len(conflict_keys)} textos duplicados con clases distintas fusionados (unión de etiquetas)")
    before = len(out)
    if near:
        kept = out.drop_duplicates("_key").set_index("_key")
        for key, row in out[out["_key"].duplicated()].set_index("_key").iterrows():
            print(f"  casi duplicado: {row['ID']} se fusiona en {kept.at[key, 'ID']} | "
                  f"{str(row['Requirement_original'])[:90]}")
    out = out.drop_duplicates("_key").drop(columns="_key").copy()
    print(f"{name}: {before - len(out)} {'casi duplicados' if near else 'duplicados exactos'} eliminados "
          f"-> {len(out)} filas")
    return out


def remove_duplicates(out: pd.DataFrame, dataset: str, merge_conflicts: bool) -> pd.DataFrame:
    """Duplicados exactos -> ID correlativos -> casi duplicados. Los ID se asignan antes de quitar
    los casi duplicados, así los requisitos que quedan conservan su ID (y los sintéticos su Anchor_ID)."""
    out = drop_duplicates(out, dataset, merge_conflicts)
    out = drop_duplicates(out, f"{dataset} (traducción idéntica)", True, spanish=True)
    out = out.reset_index(drop=True)
    out.insert(0, "ID", [f"{dataset}_{i:04d}" for i in range(1, len(out) + 1)])
    out = drop_duplicates(out, f"{dataset} (casi duplicados)", merge_conflicts, near=True)
    return drop_duplicates(out, f"{dataset} (traducción casi idéntica)", True, spanish=True, near=True)


def merge_manual_duplicates(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Casi duplicados que la clave automática no detecta: la copia eliminada cede sus etiquetas."""
    table = pd.read_csv(MANUAL_DUPLICATES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    pairs = {r.ID_eliminado: r.ID_conservado for r in table.itertuples() if r.ID_eliminado.startswith(f"{dataset}_")}
    if not pairs:
        return out
    out = out.reset_index(drop=True)
    position = {id_: i for i, id_ in enumerate(out["ID"])}
    missing = [id_ for pair in pairs.items() for id_ in pair if id_ not in position]
    if missing:
        raise ValueError(f"{dataset}: {MANUAL_DUPLICATES.name} menciona ID que no existen: {missing}")
    labels = [col for col in LABEL_COLS if col in out.columns]
    for drop, keep in pairs.items():
        i, j = position[keep], position[drop]
        out.loc[i, labels] = out.loc[[i, j], labels].astype(int).max()
        print(f"  casi duplicado (manual): {drop} se fusiona en {keep} | {str(out.at[j, 'Requirement_original'])[:90]}")
    out = out.drop(index=[position[d] for d in pairs]).reset_index(drop=True)
    print(f"{dataset}: {len(pairs)} casi duplicados manuales eliminados -> {len(out)} filas")
    return out


# =========================================================================== paso 4: subcategorías por LLM


def add_llm_subcategories(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """PROMISE y SecReq: subcategorías = voto mayoritario de 3 corridas del LLM sobre el inglés."""
    votes = pd.read_csv(LLM_LABELS, encoding="utf-8-sig")
    votes = {(r.dataset, norm_text(r.text)): r for r in votes.itertuples()}
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "structural_zero"
    out[SUBCATEGORY_COLS] = 0
    for idx in out.index[out["Security"].eq(1)]:
        vote = votes[(dataset, norm_text(out.at[idx, "Requirement_original"]))]
        for col in SUBCATEGORY_COLS:
            out.at[idx, col] = int(getattr(vote, col))
        out.at[idx, "Prelabel_evidence"] = "SON(claude-sonnet-5-5, 3 corridas): " + str(vote.rationale)[:200]
        out.at[idx, "Subcategory_label_status"] = "llm_provisional"
    return out


# =========================================================================== paso 6: revisiones manuales


def apply_translation_fixes(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Correcciones de la revisión de traducciones. Cada una guarda el español que reemplaza; si ya no
    coincide, la traducción de origen cambió y la corrección se debe revisar en vez de aplicarse."""
    fixes = pd.read_csv(TRANSLATION_FIXES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    fixes = fixes[fixes["ID"].str.startswith(f"{dataset}_")]
    position = {id_: i for i, id_ in enumerate(out["ID"])}
    stale = []
    for fix in fixes.itertuples():
        i = position.get(fix.ID)
        if i is None or out.at[i, TEXT_COL] != fix.ES_anterior:
            stale.append(fix.ID)
            continue
        out.at[i, TEXT_COL] = fix.ES_corregido
        out.at[i, "Translation_source"] += "+revisada"
    if stale:
        raise ValueError(f"{dataset}: {len(stale)} correcciones de traducción no coinciden con el texto actual "
                         f"({TRANSLATION_FIXES.name}), p. ej. {stale[:5]}")
    if len(fixes):
        print(f"{dataset}: {len(fixes)} traducciones corregidas ({TRANSLATION_FIXES.name})")
    return out


def label_abbreviation(row: pd.Series) -> str:
    return "/".join(a for a, col in LABEL_ABBREVIATIONS.items() if int(row[col]) == 1) or "NINGUNA"


def apply_label_fixes(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Correcciones de la revisión de subcategorías (solo requisitos de seguridad). Igual que con las
    traducciones, cada una guarda las etiquetas que reemplaza y se rechaza si ya no coinciden."""
    fixes = pd.read_csv(LABEL_FIXES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    fixes = fixes[fixes["ID"].str.startswith(f"{dataset}_")]
    position = {id_: i for i, id_ in enumerate(out["ID"])}
    stale = []
    for fix in fixes.itertuples():
        i = position.get(fix.ID)
        if i is None or int(out.at[i, "Security"]) != 1 or label_abbreviation(out.loc[i]) != fix.Antes:
            stale.append(fix.ID)
            continue
        new = set() if fix.Despues == "NINGUNA" else set(fix.Despues.split("/"))
        if new - set(LABEL_ABBREVIATIONS):
            raise ValueError(f"{dataset}: etiqueta desconocida en {LABEL_FIXES.name}: {fix.ID} {fix.Despues}")
        for abbreviation, col in LABEL_ABBREVIATIONS.items():
            out.at[i, col] = int(abbreviation in new)
        out.at[i, "Subcategory_label_status"] = "revision_v2"
    if stale:
        raise ValueError(f"{dataset}: {len(stale)} correcciones de etiquetas no coinciden con las actuales "
                         f"({LABEL_FIXES.name}), p. ej. {stale[:5]}")
    if len(fixes):
        print(f"{dataset}: {len(fixes)} etiquetas de subcategoría corregidas ({LABEL_FIXES.name})")
    return out


def final_cleanup(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = out.reset_index(drop=True)
    # Las comillas se quitan al final porque la traducción y las etiquetas del LLM se buscan por el inglés.
    for col in ("Requirement_original", TEXT_COL):
        cleaned = out[col].map(strip_export_quotes)
        changed = int((cleaned != out[col]).sum())
        if changed:
            print(f"{dataset}: {changed} textos de {col} sin comillas de exportación")
        out[col] = cleaned
    out = apply_translation_fixes(out, dataset)
    out = apply_label_fixes(out, dataset)
    if out[TEXT_COL].map(near_duplicate_key).duplicated().any():
        raise AssertionError(f"{dataset}: quitar las comillas dejó requisitos duplicados.")
    for col in LABEL_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    if (out["Security"].eq(0) & out[SUBCATEGORY_COLS].any(axis=1)).any():
        raise AssertionError(f"{dataset}: hay subcategorías positivas con Security=0.")
    out.insert(1, "Dataset", dataset)
    return out[OUTPUT_COLS]


# =========================================================================== paso 7: guardar


def task_tables(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(binaria, subcategorías) de un dataset, derivadas de su tabla completa.

    binaria: todos los requisitos y su etiqueta Security (como DOSSPRE 2.0Binary).
    subcategorías: solo Security=1, con las cuatro subcategorías (como DOSSPRE 3.0Multi-class).
    """
    binary = full[[*COMMON_COLS, "Security"]]
    security = full[full["Security"].eq(1)]
    subcategories = security[[*COMMON_COLS, *SUBCATEGORY_COLS, "Subcategory_label_status", "Prelabel_evidence"]]
    return binary.reset_index(drop=True), subcategories.reset_index(drop=True)


def save(full: pd.DataFrame, dataset: str) -> None:
    full_table_path(dataset).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(full_table_path(dataset), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    binary, subcategories = task_tables(full)
    binary.to_csv(task_file_path(dataset, "binary"), index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    subcategories.to_csv(task_file_path(dataset, "subcategories"), index=False, encoding="utf-8-sig",
                         quoting=csv.QUOTE_MINIMAL)
    empty = int((~subcategories[SUBCATEGORY_COLS].any(axis=1)).sum())
    print(f"{dataset}: {len(full)} filas | positivos {full[LABEL_COLS].sum().to_dict()}\n"
          f"  binaria: {len(binary)} filas; subcategorías: {len(subcategories)} filas ({empty} sin subcategoría)")


# =========================================================================== flujo por dataset


def build(dataset: str) -> pd.DataFrame:
    if dataset == "dosspre":
        table = read_dosspre()                                        # pasos 1, 2 y 4
        table = remove_duplicates(table, dataset, merge_conflicts=True)  # paso 3
    else:
        table = read_promise() if dataset == "promise" else read_secreq()  # pasos 1 y 2
        table = remove_duplicates(table, dataset, merge_conflicts=False)   # paso 3
        table = add_llm_subcategories(table, dataset)                      # paso 4
    table = merge_manual_duplicates(table, dataset)                   # paso 5
    return final_cleanup(table, dataset)                              # paso 6


def main() -> None:
    for dataset in DATASETS:
        save(build(dataset), dataset)                                 # paso 7
    print("\nVerifique la cobertura con: python scripts/verify_subcategories.py")


if __name__ == "__main__":
    main()
