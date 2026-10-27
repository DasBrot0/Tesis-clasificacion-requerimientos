#!/usr/bin/env python3
"""Construye los tres datasets de entrenamiento (uno por fuente, sin unirlos).

    datasets/original/  ->  datasets/training/{promise,secreq,dosspre}.csv

Salida multietiqueta común (5 objetivos): Security + cuatro subcategorías ISO/IEC 25010
(Non_repudiation y Accountability se descartaron el 25-10-2026 por falta de datos).

- PROMISE_exp: SE -> Security=1; las demás categorías RNF -> Security=0. Se descartan las filas
  sintéticas/prebalanceadas (ID o PROJECT originales vacíos, PROYECTO=CHATGPT).
- SecReq: sec -> Security=1; nonsec -> Security=0.
- DOSSPRE: Security sale de "2.0Binary" (SR/NSR) y la subcategoría de la clase de "1.0Original"
  (ver DOSSPRE_CLASS_MAP); Availability y Maintainability pasan a Security=0. El texto en español es la traducción DeepL + glosario (corrida
  provisional): datasets/interim/translations/deepl_provisional.jsonl.
- SecReq: texto en inglés de datasets/interim/glossary/source_en.csv + traducción DeepL.
- PROMISE: conserva el español del antecedente (Translation_source=promise_exp_translated).

Subcategorías (corrida PROVISIONAL, sin validación de expertos): PROMISE y SecReq toman el voto
mayoritario de 3 corridas de claude-sonnet-5-5 sobre el texto en INGLÉS
(datasets/interim/labels/son_prelabels.csv), estado `llm_provisional`. Las expresiones regulares
PATTERNS (`legacy_regex_labels`) ya no generan etiquetas: solo sirven para
`python scripts/prepare_datasets.py --report-regex-agreement`.
La deduplicación se hace sobre el texto en inglés.

Uso:  python scripts/prepare_datasets.py [--report-regex-agreement]
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
SOURCE_EN = INTERIM_DIR / "glossary" / "source_en.csv"
DEEPL_TRANSLATION = INTERIM_DIR / "translations" / "deepl_provisional.jsonl"
SON_LABELS = INTERIM_DIR / "labels" / "son_prelabels.csv"
TRANSLATION_SOURCE_DEEPL = "deepl+glossary_provisional"
TRANSLATION_SOURCE_PROMISE = "promise_exp_translated"
DOSSPRE_ORIGINAL = "DOSSPRE 1.0Original.csv"
DOSSPRE_BINARY = "DOSSPRE 2.0Binary.csv"

# Clase DOSSPRE (minúsculas) -> subcategoría ISO/IEC 25010.
DOSSPRE_CLASS_MAP: dict[str, Optional[str]] = {
    "confidentiality": "Confidentiality",
    "authorisation": "Confidentiality",
    "integrity": "Integrity",
    # Accountability se descartó (25-10-2026): Auditing queda con Security=1 y sin subcategoría.
    "auditing": None,
    "authentication": "Authenticity",
    "identification": "Authenticity",
    "immunity": "Resistance",
    "intrusiond": "Resistance",
    "detection": "Resistance",
    "survivability": "Resistance",
}
# SR en DOSSPRE (Kadebu et al., 2023), pero en ISO/IEC 25010 son Fiabilidad y Mantenibilidad, no
# Seguridad: se fuerzan a Security=0, igual que A y MN en PROMISE.
DOSSPRE_OUT_OF_SECURITY = {"availability", "maintainability", "maintenability", "maintanability"}
DOSSPRE_NON_SECURITY = {"functionality", "performance", "usability", "portability"}


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
            "Translation_source": TRANSLATION_SOURCE_PROMISE,
        })
    return rows


def _deepl_map() -> dict:
    """Texto EN normalizado -> traducción ES de DeepL (P6)."""
    import json
    with open(DEEPL_TRANSLATION, encoding="utf-8") as fh:
        return {_norm_text(r["en"]): r["es"] for r in map(json.loads, fh)}


def _secreq_rows() -> list:
    src = pd.read_csv(SOURCE_EN, encoding="utf-8-sig")
    src = src[src["Dataset"] == "secreq"]
    deepl = _deepl_map()
    rows = []
    for r in src.itertuples():
        rows.append({
            "Source_file": f"SecReq/{r.Spec}",
            "Project": r.Spec,
            "Source_ID": r.Source_ID,
            "Requirement_original": _clean_text(r.Requirement_en),
            TEXT_COL: _clean_text(deepl[_norm_text(r.Requirement_en)]),
            "Original_label": r.Label,
            "Security": _normalize_security_source_label(r.Label),
            "Translation_source": TRANSLATION_SOURCE_DEEPL,
        })
    return rows


# -------- LEGADO: preanotación por expresiones regulares sobre el español --------
# Ya no genera etiquetas; solo se usa en --report-regex-agreement.
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


def legacy_regex_labels(text: str) -> tuple[dict[str, int], str]:
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


def _deduplicate(out: pd.DataFrame, name: str, merge_conflicts: bool = False, by_spanish: bool = False) -> pd.DataFrame:
    """Drop exact duplicate texts.

    A duplicated text with contradictory labels is an error, unless merge_conflicts is set: then
    the group keeps the union of its labels (DOSSPRE repeats requirements across projects under
    different classes, which is a legitimate multilabel case).
    """
    # clave: inglés; con by_spanish se deduplica además la traducción (inglés distinto, mismo español)
    out["_key"] = (out[TEXT_COL] if by_spanish else out["Requirement_original"].fillna(out[TEXT_COL])).map(_norm_text)
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
               TEXT_COL, "Translation_source", "Original_label", *LABEL_COLS, "Subcategory_label_status",
               "Prelabel_evidence"]
    path = DATASETS[dataset]
    path.parent.mkdir(parents=True, exist_ok=True)
    out[ordered].to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return out[ordered]


def _son_label_map() -> dict:
    son = pd.read_csv(SON_LABELS, encoding="utf-8-sig")
    return {(r.dataset, _norm_text(r.text)): r for r in son.itertuples()}


def _build_llm_labelled(rows: list, dataset: str) -> pd.DataFrame:
    """PROMISE/SecReq: subcategorías = voto mayoritario de SON sobre el inglés (provisional)."""
    out = _deduplicate(pd.DataFrame(rows), dataset)
    out = _deduplicate(out, dataset + " (traducción idéntica)", merge_conflicts=True, by_spanish=True)
    son = _son_label_map()
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "structural_zero"
    for col in SUBCATEGORY_COLS:
        out[col] = 0
    for idx in out.index[out["Security"].eq(1)]:
        row = son[(dataset, _norm_text(out.at[idx, "Requirement_original"]))]
        for col in SUBCATEGORY_COLS:
            out.at[idx, col] = int(getattr(row, col))
        out.at[idx, "Prelabel_evidence"] = "SON(claude-sonnet-5-5, 3 corridas): " + str(row.rationale)[:200]
        out.at[idx, "Subcategory_label_status"] = "llm_provisional"
    return _finalize(out, dataset)


def build_promise() -> pd.DataFrame:
    rows = [row for name in PROMISE_FILES for row in _promise_rows(ORIGINAL_DIR / name)]
    return _build_llm_labelled(rows, "promise")


def build_secreq() -> pd.DataFrame:
    return _build_llm_labelled(_secreq_rows(), "secreq")


def _dosspre_text(value: object) -> str:
    # Los requisitos vienen entre comillas simples y con comillas tipográficas de cp1252.
    text = str(value).strip().strip("'").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"')).strip()


def _dosspre_translation(texts: pd.Series) -> pd.Series:
    """Traducción DeepL + glosario (P6) por texto en inglés."""
    deepl = _deepl_map()
    keys = texts.map(_norm_text)
    missing = sorted(set(keys) - set(deepl))
    if missing:
        raise ValueError(f"dosspre: {len(missing)} textos sin traducción DeepL, p. ej. {missing[:3]}")
    return keys.map(deepl)


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
    unknown = set(classes) - set(DOSSPRE_CLASS_MAP) - DOSSPRE_NON_SECURITY - DOSSPRE_OUT_OF_SECURITY
    if unknown:
        raise ValueError(f"Clases DOSSPRE sin mapear: {sorted(unknown)}")
    for col in SUBCATEGORY_COLS:
        out[col] = 0
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "structural_zero"
    excluded = classes.isin(DOSSPRE_OUT_OF_SECURITY) & out["Security"].eq(1)
    out.loc[excluded, "Security"] = 0
    out.loc[excluded, "Prelabel_evidence"] = "Security=0: SR en DOSSPRE, fuera de Seguridad en ISO/IEC 25010"
    for idx in out.index[out["Security"].eq(1)]:
        target = DOSSPRE_CLASS_MAP.get(classes[idx])
        evidence = []
        if target:
            out.at[idx, target] = 1
            evidence.append(f"{target}: clase DOSSPRE '{out.at[idx, 'Original_label']}'")
        out.at[idx, "Prelabel_evidence"] = " | ".join(evidence)
        out.at[idx, "Subcategory_label_status"] = "class_mapped" if evidence else "subcategory_dropped"
    out[TEXT_COL] = _dosspre_translation(out["Requirement_original"])
    out["Translation_source"] = TRANSLATION_SOURCE_DEEPL
    out = _deduplicate(out, "dosspre", merge_conflicts=True)
    out = _deduplicate(out, "dosspre (traducción idéntica)", merge_conflicts=True, by_spanish=True)
    return _finalize(out, "dosspre")


def report_regex_agreement() -> None:
    """Compara las etiquetas actuales con las de las antiguas expresiones regulares (sobre el español)."""
    for name in ("promise", "secreq"):
        df = pd.read_csv(DATASETS[name], encoding="utf-8-sig")
        sec = df[df["Security"] == 1]
        regex = pd.DataFrame([legacy_regex_labels(t)[0] for t in sec[TEXT_COL]], index=sec.index)
        print(f"\n{name}: {len(sec)} requisitos de seguridad")
        changed = 0
        for col in SUBCATEGORY_COLS:
            new, old = sec[col].astype(int), regex[col].astype(int)
            diff = int((new != old).sum())
            changed += diff
            print(f"  {col:16s} nuevo={int(new.sum()):3d} regex={int(old.sum()):3d} cambian={diff:3d} ({diff / len(sec):.1%})")
        total = len(sec) * len(SUBCATEGORY_COLS)
        print(f"  etiquetas cambiadas: {changed}/{total} ({changed / total:.1%})")


def main() -> None:
    for name, builder in (("promise", build_promise), ("secreq", build_secreq), ("dosspre", build_dosspre)):
        frame = builder()
        counts = frame[LABEL_COLS].sum().to_dict()
        print(f"{name}: {len(frame)} filas -> {DATASETS[name]}\n  positivos: {counts}")
    print("\nVerifique la cobertura con: python scripts/verify_subcategories.py")


if __name__ == "__main__":
    np.random.seed(0)
    if "--report-regex-agreement" in sys.argv:
        report_regex_agreement()
    else:
        main()
