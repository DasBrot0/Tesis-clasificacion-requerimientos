#!/usr/bin/env python3
"""Construye el dataset experimental de SEGURIDAD para la tesis.

Salida multietiqueta (7 objetivos):
    Security,
    Confidentiality, Integrity, Non_repudiation,
    Accountability, Authenticity, Resistance.

Diseño:
- PROMISE_exp_translated: SE -> Security=1; las demás categorías RNF -> Security=0.
- SecReq: sec -> Security=1; nonsec -> Security=0.
- Por defecto descarta filas sintéticas/prebalanceadas detectadas en
  Balanced_promise_exp_nfr_train (p. ej. PROYECTO=CHATGPT o ID original vacío),
  porque la metodología exige dividir train/validation/test antes del aumento.
- Elimina duplicados exactos por Requirement_ES y falla si un mismo texto tiene
  etiquetas generales de seguridad contradictorias.
- Puede crear PREANOTACIONES automáticas de las 6 subcategorías para pruebas de
  pipeline. Estas NO sustituyen el etiquetado/consenso de expertos de la tesis.

Ejemplo con los archivos actuales:
    python build_security_dataset.py \
      --promise Balanced_promise_exp_nfr_train.csv Spanish_promise_exp_nfr_test.csv \
      --secreq SecReq_es.csv \
      --output dataset_seguridad_provisional.csv \
      --prelabel-subcategories

Para el experimento final, sustituya Balanced_promise_exp_nfr_train.csv por
Spanish_promise_exp_nfr_train.csv (sin balancear) cuando esté disponible.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SECURITY_LABELS = [
    "Security",
    "Confidentiality",
    "Integrity",
    "Non_repudiation",
    "Accountability",
    "Authenticity",
    "Resistance",
]
SUBCATEGORY_LABELS = SECURITY_LABELS[1:]


def _norm_name(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _norm_text(value: object) -> str:
    s = unicodedata.normalize("NFKD", str(value).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).replace("\ufeff", " ")
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


def _promise_rows(path: Path, include_preaugmented: bool) -> list[dict]:
    df = _read_csv_flexible(path)
    original_col = _find_col(df, ["REQUIREMENT", "Requirement_original", "Requirement"])
    spanish_col = _find_col(df, ["REQUISITO", "Requirement_ES", "Requisito", "Spanish_requirement"])
    label_col = _find_col(df, ["TIPO", "LABEL", "Original_label", "Label"])
    project_col = _find_col(df, ["PROJECT", "Project"])
    translated_project_col = _find_col(df, ["PROYECTO", "Proyecto"])
    id_col = _find_col(df, ["ID", "Source_ID"])

    if spanish_col is None:
        raise ValueError(f"{path}: falta columna de texto español (REQUISITO/Requirement_ES).")
    if label_col is None:
        raise ValueError(f"{path}: falta columna TIPO/LABEL.")

    partition = "test" if "test" in path.name.lower() else ("train" if "train" in path.name.lower() else "unknown")
    rows: list[dict] = []
    for idx, row in df.iterrows():
        req_es = _clean_text(row[spanish_col])
        if not req_es:
            continue
        sec = _normalize_security_source_label(row[label_col])
        if sec is None:
            continue
        is_synth = _is_preaugmented_promise_row(row, id_col, project_col, translated_project_col)
        if is_synth and not include_preaugmented:
            continue
        rows.append({
            "Source": "PROMISE_exp_translated",
            "Source_file": path.name,
            "Source_partition": partition,
            "Project": _clean_text(row[project_col]) if project_col else _clean_text(row[translated_project_col]) if translated_project_col else None,
            "Source_ID": _clean_text(row[id_col]) if id_col else str(idx),
            "Requirement_original": _clean_text(row[original_col]) if original_col else None,
            "Requirement_ES": req_es,
            "Original_label": _clean_text(row[label_col]),
            "Security": sec,
            "Is_synthetic": bool(is_synth),
            "General_label_status": "source_original" if not is_synth else "source_synthetic",
        })
    return rows


def _secreq_rows(path: Path) -> list[dict]:
    df = _read_csv_flexible(path)
    text_col = _find_col(df, ["Requirement_ES", "Requisito", "Requerimiento", "Requirement", "Texto"])
    label_col = _find_col(df, ["etiqueta", "Classification", "Clasificacion", "Clasificación", "Label", "Tipo"])
    id_col = _find_col(df, ["ID", "Source_ID"])
    project_col = _find_col(df, ["Project", "Proyecto", "Specification", "Especificacion"])

    if text_col is None or label_col is None:
        raise ValueError(f"{path}: se requieren columnas de texto y etiqueta sec/nonsec.")

    rows: list[dict] = []
    for idx, row in df.iterrows():
        req_es = _clean_text(row[text_col])
        if not req_es:
            continue
        sec = _normalize_security_source_label(row[label_col])
        if sec is None:
            continue
        # SecReq_es consolidado actual perdió la especificación de origen; se conserva SecReq como proyecto genérico.
        project = _clean_text(row[project_col]) if project_col else "SecReq"
        rows.append({
            "Source": "SecReq",
            "Source_file": path.name,
            "Source_partition": "unsplit",
            "Project": project,
            "Source_ID": _clean_text(row[id_col]) if id_col else str(idx + 1),
            "Requirement_original": None,
            "Requirement_ES": req_es,
            "Original_label": _clean_text(row[label_col]),
            "Security": sec,
            "Is_synthetic": False,
            "General_label_status": "source_original",
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


def _coerce_existing_subcats(df: pd.DataFrame) -> pd.DataFrame:
    for col in SUBCATEGORY_LABELS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        invalid = df[col].dropna()[~df[col].dropna().isin([0, 1])]
        if not invalid.empty:
            raise ValueError(f"{col}: hay valores distintos de 0/1.")
    return df


def build_security_dataset(
    promise_paths: list[Path],
    secreq_paths: list[Path],
    output_path: Path,
    include_preaugmented: bool = False,
    prelabel_subcategories: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in promise_paths:
        rows.extend(_promise_rows(path, include_preaugmented=include_preaugmented))
    for path in secreq_paths:
        rows.extend(_secreq_rows(path))
    if not rows:
        raise ValueError("No se generaron registros.")

    out = pd.DataFrame(rows)
    out["Duplicate_key"] = out["Requirement_ES"].map(_norm_text)

    # Antes de deduplicar, verificar que el mismo texto no tenga etiquetas generales contradictorias.
    conflicts = out.groupby("Duplicate_key")["Security"].nunique()
    bad_keys = set(conflicts[conflicts > 1].index)
    if bad_keys:
        examples = out[out["Duplicate_key"].isin(list(bad_keys))][["Requirement_ES", "Source", "Original_label", "Security"]]
        raise ValueError("Hay textos duplicados con etiquetas Security contradictorias:\n" + examples.head(20).to_string(index=False))

    before = len(out)
    # Preferir registros no sintéticos y con texto original disponible.
    out["_orig_rank"] = out["Requirement_original"].notna().astype(int)
    out = out.sort_values(["Duplicate_key", "Is_synthetic", "_orig_rank"], ascending=[True, True, False])
    out = out.drop_duplicates("Duplicate_key", keep="first").copy()
    duplicates_removed = before - len(out)
    out.drop(columns=["_orig_rank"], inplace=True)

    out = _coerce_existing_subcats(out)
    out["Prelabel_evidence"] = ""
    out["Subcategory_label_status"] = "pending_review"

    # Negativos de seguridad -> todas las subcategorías necesariamente 0.
    neg = out["Security"].eq(0)
    out.loc[neg, SUBCATEGORY_LABELS] = 0
    out.loc[neg, "Subcategory_label_status"] = "structural_zero"

    # Positivos: si no hay etiquetas, opcionalmente crear preanotación automática trazable.
    pos = out["Security"].eq(1)
    for idx in out.index[pos]:
        existing = out.loc[idx, SUBCATEGORY_LABELS]
        if existing.notna().all():
            out.at[idx, "Subcategory_label_status"] = "provided"
            continue
        if prelabel_subcategories:
            labels, evidence = _prelabel(out.at[idx, "Requirement_ES"])
            for col, value in labels.items():
                out.at[idx, col] = int(value)
            out.at[idx, "Prelabel_evidence"] = evidence
            out.at[idx, "Subcategory_label_status"] = "auto_prelabel"

    out["Security"] = pd.to_numeric(out["Security"], errors="raise").astype("Int64")
    for col in SUBCATEGORY_LABELS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["Ready_for_training"] = out[SECURITY_LABELS].notna().all(axis=1)
    out["Needs_expert_review"] = out["Security"].eq(1) & ~out["Subcategory_label_status"].isin(["expert_validated", "consensus"])

    # ID estable después de la deduplicación.
    out = out.sort_values(["Source", "Source_file", "Source_ID"], kind="stable").reset_index(drop=True)
    out.insert(0, "ID", np.arange(1, len(out) + 1, dtype=int))
    out.drop(columns=["Duplicate_key"], inplace=True)

    ordered = [
        "ID", "Source", "Source_file", "Source_partition", "Project", "Source_ID",
        "Requirement_original", "Requirement_ES", "Original_label", "Is_synthetic",
        *SECURITY_LABELS,
        "General_label_status", "Subcategory_label_status", "Ready_for_training",
        "Needs_expert_review", "Prelabel_evidence",
    ]
    out = out[ordered]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # Cola de revisión: solo positivos de seguridad, porque los negativos tienen ceros estructurales.
    review_path = output_path.with_name(output_path.stem + "_review_security_positive.csv")
    out[out["Security"].eq(1)].to_csv(review_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary_path = output_path.with_suffix(".summary.txt")
    summary = [
        f"Output: {output_path}",
        f"Total registros únicos: {len(out)}",
        f"Duplicados exactos eliminados: {duplicates_removed}",
        f"Security=1: {int(out['Security'].sum())}",
        f"Security=0: {int((out['Security'] == 0).sum())}",
        f"Listos para entrenamiento: {int(out['Ready_for_training'].sum())}",
        f"Requieren revisión experta de subcategorías: {int(out['Needs_expert_review'].sum())}",
        "",
        "Distribución de subcategorías (preanotación/proporcionadas):",
    ]
    for col in SUBCATEGORY_LABELS:
        summary.append(f"{col}: {int(out[col].fillna(0).sum())}")
    summary.extend([
        "",
        "IMPORTANTE: las filas con Subcategory_label_status=auto_prelabel son preanotaciones para pruebas del pipeline, no ground truth final.",
        "Para resultados de tesis, sustituirlas por etiquetas expert_validated/consensus.",
    ])
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promise", nargs="+", default=[], help="CSV(s) PROMISE traducidos")
    parser.add_argument("--secreq", nargs="+", default=[], help="CSV(s) SecReq traducidos")
    parser.add_argument("--output", default="dataset_seguridad.csv", help="CSV de salida")
    parser.add_argument(
        "--include-preaugmented",
        action="store_true",
        help="Incluye filas sintéticas/prebalanceadas ya presentes en PROMISE. No recomendado para el experimento final.",
    )
    parser.add_argument(
        "--prelabel-subcategories",
        action="store_true",
        help="Genera preanotación heurística de las 6 subcategorías para pruebas. Requiere revisión de expertos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    promise_paths = [Path(p) for p in args.promise]
    secreq_paths = [Path(p) for p in args.secreq]
    missing = [str(p) for p in promise_paths + secreq_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Archivos no encontrados: {missing}")
    if not promise_paths and not secreq_paths:
        raise ValueError("Debe indicar al menos un archivo --promise o --secreq")

    df = build_security_dataset(
        promise_paths=promise_paths,
        secreq_paths=secreq_paths,
        output_path=Path(args.output),
        include_preaugmented=args.include_preaugmented,
        prelabel_subcategories=args.prelabel_subcategories,
    )
    print(df.head().to_string(index=False))
    print(f"\nDataset guardado en: {args.output}")
    print(f"Resumen: {Path(args.output).with_suffix('.summary.txt')}")
    print(f"Cola de revisión: {Path(args.output).with_name(Path(args.output).stem + '_review_security_positive.csv')}")


if __name__ == "__main__":
    main()
