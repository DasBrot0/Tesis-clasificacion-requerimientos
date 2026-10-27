#!/usr/bin/env python3
"""Construye los tres datasets de entrenamiento (uno por fuente, sin unirlos).

    datasets/original/  ->  datasets/training/{promise,secreq,dosspre}.csv

Salida multietiqueta común (7 objetivos): Security + seis subcategorías ISO/IEC 25010.

- PROMISE_exp: SE -> Security=1; las demás categorías RNF -> Security=0. Se descartan las filas
  sintéticas/prebalanceadas (ID o PROJECT originales vacíos, PROYECTO=CHATGPT).
- SecReq: sec -> Security=1; nonsec -> Security=0.
- DOSSPRE: Security sale de "2.0Binary" (SR/NSR) y la subcategoría de la clase de "1.0Original"
  (ver DOSSPRE_CLASS_MAP). El texto se traduce al español con Helsinki-NLP/opus-mt-en-es porque
  BETO y RoBERTa-BNE son modelos en español; la traducción se guarda en datasets/interim/.

PROMISE y SecReq no traen subcategorías: se preanotan con las expresiones regulares PATTERNS
(`auto_prelabel`). Esas preanotaciones NO sustituyen el etiquetado/consenso de expertos.

Uso:  python scripts/prepare_datasets.py
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    DATASETS, INTERIM_DIR, LABEL_COLS, ORIGINAL_DIR, SUBCATEGORY_COLS, TEXT_COL,
)

PROMISE_FILES = ["Balanced_promise_exp_nfr_train.csv", "Spanish_promise_exp_nfr_test_recovered.csv"]
SECREQ_FILES = ["SecReq_es.csv"]
DOSSPRE_ORIGINAL = "DOSSPRE 1.0Original.csv"
DOSSPRE_BINARY = "DOSSPRE 2.0Binary.csv"
DOSSPRE_TRANSLATION = INTERIM_DIR / "dosspre_es_translation.csv"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"

# Clase DOSSPRE (minúsculas) -> subcategoría ISO/IEC 25010. None = requisito de seguridad según
# DOSSPRE (SR) sin equivalente entre las seis subcategorías; queda con Security=1 y el resto en 0.
DOSSPRE_CLASS_MAP: dict[str, Optional[str]] = {
    "confidentiality": "Confidentiality",
    "authorisation": "Confidentiality",
    "integrity": "Integrity",
    "auditing": "Accountability",
    "authentication": "Authenticity",
    "identification": "Authenticity",
    "immunity": "Resistance",
    "intrusiond": "Resistance",
    "detection": "Resistance",
    "survivability": "Resistance",
    "availability": None,
    "maintainability": None,
    "maintenability": None,
    "maintanability": None,
}
DOSSPRE_NON_SECURITY = {"functionality", "performance", "usability", "portability"}
# DOSSPRE no tiene clase de no repudio; solo se marca ante una mención explícita.
DOSSPRE_NON_REPUDIATION = re.compile(r"non[- ]?repudiat|repudiat", re.IGNORECASE)


def _norm_name(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _norm_text(value: object) -> str:
    s = unicodedata.normalize("NFKD", str(value).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip()


def _clean_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).replace("﻿", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _find_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    normalized = {_norm_name(c): c for c in df.columns}
    for alias in aliases:
        if _norm_name(alias) in normalized:
            return normalized[_norm_name(alias)]
    return None


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts = [
        ("utf-8-sig", ";"), ("utf-8-sig", ","),
        ("cp1252", ";"), ("cp1252", ","),
        ("latin1", ";"), ("latin1", ","),
    ]
    errors = []
    for encoding, sep in attempts:
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            errors.append(f"{encoding}/{sep}: {exc}")
    raise ValueError(f"No se pudo leer {path}. Intentos: {' | '.join(errors[-4:])}")


def _normalize_security_source_label(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    s = _norm_name(value)
    if s in {"se", "sec", "security", "seguridad", "1", "true"}:
        return 1
    if s in {"nonsec", "nonsecurity", "noseguridad", "0", "false"}:
        return 0
    # En PROMISE, cualquier otra etiqueta RNF actúa como negativo para la tarea Seguridad/No seguridad.
    if s in {"a", "po", "l", "ft", "sc", "mn", "lf", "pe", "o", "us"}:
        return 0
    return None


def _is_preaugmented_promise_row(row: pd.Series, id_col: Optional[str], project_col: Optional[str], translated_project_col: Optional[str]) -> bool:
    # El dataset balanceado publicado contiene ejemplos generados con ChatGPT con ID/PROJECT originales vacíos.
    if id_col and pd.isna(row[id_col]):
        return True
    if project_col and pd.isna(row[project_col]):
        return True
    if translated_project_col:
        p = _clean_text(row[translated_project_col])
        if p and _norm_name(p) in {"chatgpt", "synthetic", "sintetico", "sintetica"}:
            return True
    return False


def _promise_rows(path: Path) -> list[dict]:
    df = _read_csv_flexible(path)
    original_col = _find_col(df, ["REQUIREMENT", "Requirement_original", "Requirement"])
    spanish_col = _find_col(df, ["REQUISITO", "Requirement_ES", "Requisito", "Spanish_requirement"])
    label_col = _find_col(df, ["TIPO", "LABEL", "Original_label", "Label"])
    project_col = _find_col(df, ["PROJECT", "Project"])
    translated_project_col = _find_col(df, ["PROYECTO", "Proyecto"])
    id_col = _find_col(df, ["ID", "Source_ID"])
    if spanish_col is None or label_col is None:
        raise ValueError(f"{path}: faltan las columnas REQUISITO y/o TIPO.")

    rows: list[dict] = []
    for idx, row in df.iterrows():
        req_es = _clean_text(row[spanish_col])
        sec = _normalize_security_source_label(row[label_col])
        if not req_es or sec is None:
            continue
        if _is_preaugmented_promise_row(row, id_col, project_col, translated_project_col):
            continue
        rows.append({
            "Source_file": path.name,
            "Project": _clean_text(row[project_col]) if project_col else None,
            "Source_ID": _clean_text(row[id_col]) if id_col else str(idx),
            "Requirement_original": _clean_text(row[original_col]) if original_col else None,
            TEXT_COL: req_es,
            "Original_label": _clean_text(row[label_col]),
            "Security": sec,
        })
    return rows


def _secreq_rows(path: Path) -> list[dict]:
    df = _read_csv_flexible(path)
    text_col = _find_col(df, ["Requirement_ES", "Requisito", "Requerimiento", "Requirement", "Texto"])
    label_col = _find_col(df, ["etiqueta", "Classification", "Clasificacion", "Label", "Tipo"])
    if text_col is None or label_col is None:
        raise ValueError(f"{path}: se requieren columnas de texto y etiqueta sec/nonsec.")

    rows: list[dict] = []
    for idx, row in df.iterrows():
        req_es = _clean_text(row[text_col])
        sec = _normalize_security_source_label(row[label_col])
        if not req_es or sec is None:
            continue
        rows.append({
            "Source_file": path.name,
            "Project": "SecReq",
            "Source_ID": str(idx + 1),
            "Requirement_original": None,
            TEXT_COL: req_es,
            "Original_label": _clean_text(row[label_col]),
            "Security": sec,
        })
    return rows


# -------- Preanotación transparente de subcategorías --------
# Sirve para pruebas de implementación. No reemplaza la anotación por expertos.
PATTERNS: dict[str, list[str]] = {
    "Confidentiality": [
        r"confidencial", r"privacidad", r"informacion privada", r"datos privados", r"divulg", r"revel",
        r"solo .*?(?:acced|ver|consult|usar|modific|actualiz|realiz|ingresar|buscar|transmit)",
        r"unicamente .*?(?:acced|ver|consult|usar|modific|actualiz|realiz|ingresar|buscar|transmit)",
        r"unicas? personas .*?acceso", r"unico usuario autorizado", r"usuarios? no autoriz", r"entidades? no autoriz",
        r"personal autorizado", r"usuarios? autoriz", r"entidades? autoriz", r"remitente es un administrador",
        r"acceso .*?autoriz", r"acceso .*?limit", r"restriccion(?:es)? de acceso", r"derechos? de acceso",
        r"permisos?", r"privilegios?", r"control de acceso", r"visibilidad .*?limit",
        r"cifrad", r"encript", r"ssl\b", r"canal seguro", r"comunicacion segura",
        r"almacenamiento seguro", r"almacenar.*?segur", r"base de datos segura", r"base de datos protegida",
        r"no .*?mostrar.*?contrasena", r"no .*?mostrar.*?tarjeta", r"separacion .*?claves?",
        r"protegidos? frente al acceso no autorizado", r"proteger.*?divulgacion no autorizada",
    ],
    "Integrity": [
        r"integridad", r"sin alteraciones", r"no hayan sido alterad", r"manipul", r"corrup",
        r"datos? valid", r"datos? invalid", r"datos? correct", r"datos .*?correspondan", r"misma informacion",
        r"verificar.*?mac", r"validar.*?mac", r"\bmac\b", r"\bhash\b",
        r"firma(?:s)? digital", r"firma(?:s)? de .*?transaccion", r"firmad[oa]", r"firmar",
        r"coherente", r"conciliable", r"cambios .*?autoriz", r"actualiz.*?autoriz",
        r"transmision correcta", r"validacion", r"verificacion de firmas", r"verificacion de .*?certificad",
    ],
    "Non_repudiation": [
        r"no[- ]?repudio", r"repudio", r"evitar el repudio", r"prueba de que", r"comprobante",
        r"evidencia", r"disputa", r"reembolso", r"liquidar", r"resolucion de disputas",
    ],
    "Accountability": [
        r"auditor", r"trazab", r"\blogs?\b", r"registrar", r"registrad", r"registro de .*?transaccion",
        r"informe de usuario", r"marca de tiempo", r"numero de secuencia", r"contador de transacciones",
        r"identificador unico", r"identificacion unica", r"historial", r"repositorio .*?errores",
        r"seguimiento de transacciones", r"registro de eventos",
    ],
    "Authenticity": [
        r"autentic", r"identidad", r"identificacion", r"iniciar sesion", r"inicio de sesion", r"login",
        r"nombre de usuario", r"contrasena", r"\bpin\b", r"credencial", r"certificad", r"genuin",
        r"clave publica", r"clave privada", r"autoridad de certificacion", r"autenticidad del origen",
        r"verificacion del titular", r"verificacion de pin",
    ],
    "Resistance": [
        r"ataque", r"malicios", r"abuso intencional", r"abusos intencionales", r"firewall", r"antivirus", r"amenaza",
        r"reutiliz.*?credencial", r"reproduccion", r"replay", r"bloquear", r"bloquead", r"deshabilit",
        r"estado seguro", r"entorno seguro", r"seguro y certificado", r"no pueda ser elud", r"no pueden ser elud",
        r"sortead", r"fallo de autenticacion", r"multiples intentos", r"intentos repetid", r"inactividad",
        r"expiracion", r"recuperacion de memoria", r"impedir el acceso", r"rechazar .*?solicitud",
        r"cerrar .*?sesion", r"cierre .*?sesion", r"descartar .*?mensaje", r"protegerse", r"proteccion antivirus",
    ],
}


def _prelabel(text: str) -> tuple[dict[str, int], str]:
    norm = _norm_text(text)
    out: dict[str, int] = {}
    evidence: list[str] = []
    for label, pats in PATTERNS.items():
        matches = []
        for pat in pats:
            m = re.search(pat, norm)
            if m:
                matches.append(m.group(0)[:80])
        out[label] = int(bool(matches))
        if matches:
            evidence.append(f"{label}: {', '.join(matches[:3])}")
    return out, " | ".join(evidence)


def _deduplicate(out: pd.DataFrame, name: str, merge_conflicts: bool = False) -> pd.DataFrame:
    """Drop exact duplicate texts.

    A duplicated text with contradictory labels is an error, unless merge_conflicts is set: then
    the group keeps the union of its labels (DOSSPRE repeats requirements across projects under
    different classes, which is a legitimate multilabel case).
    """
    out["_key"] = out[TEXT_COL].map(_norm_text)
    label_cols = [col for col in LABEL_COLS if col in out.columns]
    conflicts = out.groupby("_key")[label_cols].nunique().max(axis=1)
    bad_keys = set(conflicts[conflicts > 1].index)
    if bad_keys and not merge_conflicts:
        examples = out[out["_key"].isin(bad_keys)][[TEXT_COL, "Original_label", *label_cols]]
        raise ValueError(f"{name}: textos duplicados con etiquetas contradictorias:\n"
                         + examples.head(20).to_string(index=False))
    if bad_keys:
        grouped = out.groupby("_key", sort=False)
        out[label_cols] = grouped[label_cols].transform("max")
        for col in ("Original_label", "Prelabel_evidence"):
            out[col] = grouped[col].transform(lambda part: " | ".join(dict.fromkeys(v for v in part if v)))
        out.loc[out["_key"].isin(bad_keys), "Subcategory_label_status"] = "class_mapped_merged_duplicates"
        print(f"{name}: {len(bad_keys)} textos duplicados con clases distintas fusionados (unión de etiquetas)")
    before = len(out)
    out = out.drop_duplicates("_key", keep="first").drop(columns="_key").copy()
    print(f"{name}: {before - len(out)} duplicados exactos eliminados -> {len(out)} filas")
    return out


def _finalize(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = out.reset_index(drop=True)
    for col in LABEL_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    if (out["Security"].eq(0) & out[SUBCATEGORY_COLS].any(axis=1)).any():
        raise AssertionError(f"{dataset}: hay subcategorías positivas con Security=0.")
    out.insert(0, "ID", [f"{dataset}_{i:04d}" for i in range(1, len(out) + 1)])
    out.insert(1, "Dataset", dataset)
    ordered = ["ID", "Dataset", "Source_file", "Project", "Source_ID", "Requirement_original",
               TEXT_COL, "Original_label", *LABEL_COLS, "Subcategory_label_status",
               "Prelabel_evidence"]
    path = DATASETS[dataset]
    path.parent.mkdir(parents=True, exist_ok=True)
    out[ordered].to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return out[ordered]


def _build_prelabelled(rows: list[dict], dataset: str) -> pd.DataFrame:
    out = _deduplicate(pd.DataFrame(rows), dataset)
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "structural_zero"
    for col in SUBCATEGORY_COLS:
        out[col] = 0
    for idx in out.index[out["Security"].eq(1)]:
        labels, evidence = _prelabel(out.at[idx, TEXT_COL])
        for col, value in labels.items():
            out.at[idx, col] = int(value)
        out.at[idx, "Prelabel_evidence"] = evidence
        out.at[idx, "Subcategory_label_status"] = "auto_prelabel"
    return _finalize(out, dataset)


def build_promise() -> pd.DataFrame:
    rows = [row for name in PROMISE_FILES for row in _promise_rows(ORIGINAL_DIR / name)]
    return _build_prelabelled(rows, "promise")


def build_secreq() -> pd.DataFrame:
    rows = [row for name in SECREQ_FILES for row in _secreq_rows(ORIGINAL_DIR / name)]
    return _build_prelabelled(rows, "secreq")


def _dosspre_text(value: object) -> str:
    # Los requisitos vienen entre comillas simples y con comillas tipográficas de cp1252.
    text = str(value).strip().strip("'").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"')).strip()


def translate_en_es(texts: list[str], batch_size: int = 32) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(device).eval()
    translated: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
        with torch.no_grad():
            generated = model.generate(**batch, num_beams=4, max_length=512)
        translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"  traducidos {len(translated)}/{len(texts)}", flush=True)
    return translated


def _dosspre_translation(texts: pd.Series) -> pd.Series:
    """English -> Spanish, cached so the training CSV can be rebuilt without re-translating."""
    cache: dict[str, str] = {}
    if DOSSPRE_TRANSLATION.exists():
        stored = pd.read_csv(DOSSPRE_TRANSLATION, encoding="utf-8-sig")
        cache = dict(zip(stored["Requirement_original"], stored[TEXT_COL]))
    missing = [text for text in dict.fromkeys(texts) if text not in cache]
    if missing:
        print(f"dosspre: traduciendo {len(missing)} requisitos con {TRANSLATION_MODEL}")
        cache.update(zip(missing, translate_en_es(missing)))
        DOSSPRE_TRANSLATION.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Requirement_original": list(cache), TEXT_COL: list(cache.values()),
                      "Translation_model": TRANSLATION_MODEL}).to_csv(
            DOSSPRE_TRANSLATION, index=False, encoding="utf-8-sig")
    return texts.map(cache)


def build_dosspre() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_DIR / DOSSPRE_ORIGINAL, encoding="cp1252")
    binary = pd.read_csv(ORIGINAL_DIR / DOSSPRE_BINARY, encoding="cp1252")
    if len(original) != len(binary) or not original["Project ID"].equals(binary["Project ID"]):
        raise ValueError("DOSSPRE 1.0 y 2.0 no están alineados fila a fila.")
    out = pd.DataFrame({
        "Source_file": DOSSPRE_ORIGINAL,
        "Project": original["Project ID"].astype(str),
        "Source_ID": (original.index + 1).astype(str),
        "Requirement_original": original["Requirement"].map(_dosspre_text),
        "Original_label": original["Class"].str.strip(),
        "Security": binary["Class"].str.strip().map({"SR": 1, "NSR": 0}),
    })
    if out["Security"].isna().any():
        raise ValueError("DOSSPRE 2.0Binary contiene clases distintas de SR/NSR.")
    classes = out["Original_label"].str.lower()
    unknown = set(classes) - set(DOSSPRE_CLASS_MAP) - DOSSPRE_NON_SECURITY
    if unknown:
        raise ValueError(f"Clases DOSSPRE sin mapear: {sorted(unknown)}")
    for col in SUBCATEGORY_COLS:
        out[col] = 0
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "structural_zero"
    for idx in out.index[out["Security"].eq(1)]:
        target = DOSSPRE_CLASS_MAP.get(classes[idx])
        evidence = []
        if target:
            out.at[idx, target] = 1
            evidence.append(f"{target}: clase DOSSPRE '{out.at[idx, 'Original_label']}'")
        match = DOSSPRE_NON_REPUDIATION.search(out.at[idx, "Requirement_original"])
        if match:
            out.at[idx, "Non_repudiation"] = 1
            evidence.append(f"Non_repudiation: {match.group(0)}")
        out.at[idx, "Prelabel_evidence"] = " | ".join(evidence)
        out.at[idx, "Subcategory_label_status"] = "class_mapped" if evidence else "no_iso25010_equivalent"
    out[TEXT_COL] = _dosspre_translation(out["Requirement_original"])
    return _finalize(_deduplicate(out, "dosspre", merge_conflicts=True), "dosspre")


def main() -> None:
    for name, builder in (("promise", build_promise), ("secreq", build_secreq), ("dosspre", build_dosspre)):
        frame = builder()
        counts = frame[LABEL_COLS].sum().to_dict()
        print(f"{name}: {len(frame)} filas -> {DATASETS[name]}\n  positivos: {counts}")
    print("\nVerifique la cobertura con: python scripts/verify_subcategories.py")


if __name__ == "__main__":
    np.random.seed(0)
    main()
