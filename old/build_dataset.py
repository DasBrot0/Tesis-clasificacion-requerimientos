#!/usr/bin/env python3
"""Construye el dataset consolidado PROMISE_exp_translated + SecReq.

El script implementa únicamente el nivel GENERAL de clasificación multietiqueta:
Security, Usability o Both. No crea todavía las 14 subcategorías.

Principios de diseño:
- conserva trazabilidad (fuente, proyecto, texto original y etiqueta original);
- exige texto en español para entrenamiento, salvo --allow-untranslated;
- NO convierte SecReq `nonsec` a Usability, porque `nonsec` solo significa
  "no identificado como seguridad" en el dataset original;
- usa una etiqueta final explícita si el archivo SecReq ya la contiene;
- puede leer tanto CSV normales como el formato histórico SecReq `texto;sec,,,,`.

Ejemplo:
    python build_dataset.py \
      --promise Spanish_promise_exp_nfr_train.csv Spanish_promise_exp_nfr_test.csv \
      --secreq CPN_traducido.csv ePurse-selective_traducido.csv GPS_traducido.csv \
      --output dataset_consolidado.csv
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

GENERAL_LABELS = {"Security", "Usability", "Both"}


def _norm_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _find_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    normalized = {_norm_name(c): c for c in df.columns}
    for alias in aliases:
        key = _norm_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def _clean_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).replace("\ufeff", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _normalize_general_label(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = _norm_name(value)
    mapping = {
        "security": "Security",
        "seguridad": "Security",
        "se": "Security",
        "sec": "Security",
        "usability": "Usability",
        "usabilidad": "Usability",
        "us": "Usability",
        "both": "Both",
        "ambos": "Both",
        "securityusability": "Both",
        "seguridadusabilidad": "Both",
        "usabilitysecurity": "Both",
        "usabilidadseguridad": "Both",
    }
    return mapping.get(s)


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Lee CSV/CSV-like usando encodings y separadores habituales."""
    attempts = [
        ("utf-8-sig", ";"),
        ("utf-8-sig", ","),
        ("cp1252", ";"),
        ("cp1252", ","),
        ("latin1", ";"),
        ("latin1", ","),
    ]
    errors = []
    for encoding, sep in attempts:
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
            # Aceptamos si hay al menos 2 columnas; un único campo suele indicar separador incorrecto.
            if df.shape[1] >= 2:
                return df
        except Exception as exc:  # pragma: no cover - diagnóstico
            errors.append(f"{encoding}/{sep}: {exc}")
    raise ValueError(f"No se pudo leer {path}. Intentos: {' | '.join(errors[-4:])}")


def _read_legacy_secreq(path: Path) -> pd.DataFrame:
    """Parser para el formato `requirement;sec,,,,` con posibles saltos internos."""
    raw = None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError(f"No se pudo decodificar {path}")

    records = []
    buffer: list[str] = []
    label_pattern = re.compile(r";\s*(sec|nonsec)\b", flags=re.IGNORECASE)
    for physical_line in raw.splitlines():
        buffer.append(physical_line.strip())
        joined = " ".join(buffer)
        match = label_pattern.search(joined)
        if not match:
            continue
        requirement = joined[: match.start()].strip().strip('"')
        label = match.group(1).lower()
        records.append({"Requirement_original": requirement, "Original_label": label})
        buffer = []

    if buffer:
        raise ValueError(
            f"Quedó un registro SecReq sin etiqueta al final de {path}: {' '.join(buffer)[:120]}"
        )
    if not records:
        raise ValueError(f"No se reconocieron registros SecReq en {path}")
    return pd.DataFrame(records)


def _read_secreq_any(path: Path) -> pd.DataFrame:
    try:
        df = _read_csv_flexible(path)
        # Si parece tabla real, se devuelve. Si no contiene columnas reconocibles, cae al legacy.
        recognizable = any(
            _find_col(df, aliases)
            for aliases in [
                ["Requirement", "Requerimiento", "Requirement_original"],
                ["Requirement_ES", "Requisito", "Traduccion", "Traducción"],
                ["Classification", "Clasificacion", "Clasificación", "Label", "Tipo"],
            ]
        )
        if recognizable:
            return df
    except Exception:
        pass
    return _read_legacy_secreq(path)


def _label_to_binary(label: Optional[str]):
    if label == "Security":
        return 1, 0
    if label == "Usability":
        return 0, 1
    if label == "Both":
        return 1, 1
    return pd.NA, pd.NA


def _promise_rows(path: Path, allow_untranslated: bool) -> list[dict]:
    df = _read_csv_flexible(path)
    original_col = _find_col(df, ["REQUIREMENT", "Requirement_original", "Requirement", "Requerimiento"])
    spanish_col = _find_col(df, ["REQUISITO", "Requirement_ES", "Requisito_ES", "Spanish_requirement"])
    label_col = _find_col(df, ["TIPO", "LABEL", "Original_label", "Label"])
    project_col = _find_col(df, ["PROYECTO", "PROJECT", "Project"])
    id_col = _find_col(df, ["ID_NEW", "ID", "Id"])
    final_col = _find_col(
        df,
        [
            "Final_classification",
            "Clasificacion_final",
            "Clasificación_final",
            "General_classification",
            "Classification_general",
        ],
    )

    if original_col is None and spanish_col is None:
        raise ValueError(f"{path}: no se encontró columna de requerimiento.")
    if label_col is None and final_col is None:
        raise ValueError(f"{path}: no se encontró etiqueta original ni clasificación final.")

    rows = []
    for idx, row in df.iterrows():
        req_original = _clean_text(row[original_col]) if original_col else None
        req_es = _clean_text(row[spanish_col]) if spanish_col else None
        if req_es is None:
            if allow_untranslated:
                req_es = req_original
            else:
                raise ValueError(
                    f"{path}: falta traducción al español en la fila {idx}. "
                    "Use --allow-untranslated solo para pruebas."
                )

        original_label = _clean_text(row[label_col]) if label_col else None
        final_label = _normalize_general_label(row[final_col]) if final_col else None
        if final_label is None:
            final_label = _normalize_general_label(original_label)

        # PROMISE completo contiene otras RNF. En este dataset solo entran SE/US o registros ya reetiquetados.
        if final_label not in GENERAL_LABELS:
            continue

        security, usability = _label_to_binary(final_label)
        rows.append(
            {
                "Source": "PROMISE_exp_translated",
                "Project": _clean_text(row[project_col]) if project_col else None,
                "Source_ID": _clean_text(row[id_col]) if id_col else str(idx),
                "Requirement_original": req_original,
                "Requirement_ES": req_es,
                "Original_label": original_label,
                "Classification": final_label,
                "Security": security,
                "Usability": usability,
                "Label_status": "ready",
            }
        )
    return rows


def _secreq_rows(path: Path, allow_untranslated: bool) -> list[dict]:
    df = _read_secreq_any(path)

    original_col = _find_col(
        df,
        ["Requirement_original", "Requirement", "Requerimiento_original", "Requerimiento", "Text", "Texto"],
    )
    spanish_col = _find_col(
        df,
        [
            "Requirement_ES",
            "Requisito",
            "Requisito_ES",
            "Requerimiento_ES",
            "Traduccion",
            "Traducción",
            "Spanish_requirement",
        ],
    )
    original_label_col = _find_col(
        df, ["Original_label", "Security_label", "SecReq_label", "Label", "Tipo", "Class"]
    )
    final_col = _find_col(
        df,
        [
            "Final_classification",
            "Clasificacion_final",
            "Clasificación_final",
            "General_classification",
            "Classification_general",
            "Final_label",
            "Etiqueta_final",
        ],
    )
    # `Classification`/`Clasificación` puede ser final (Security/Usability/Both) o original (sec/nonsec).
    ambiguous_class_col = _find_col(df, ["Classification", "Clasificacion", "Clasificación"])
    project_col = _find_col(df, ["Project", "Proyecto", "Specification", "Especificacion", "Especificación"])
    id_col = _find_col(df, ["ID", "Id", "Source_ID"])

    # Formato legacy ya normalizado por _read_legacy_secreq.
    if "Requirement_original" in df.columns and "Original_label" in df.columns:
        original_col = "Requirement_original"
        original_label_col = "Original_label"

    if original_col is None and spanish_col is None:
        raise ValueError(f"{path}: no se encontró columna de requerimiento SecReq.")

    rows = []
    for idx, row in df.iterrows():
        req_original = _clean_text(row[original_col]) if original_col else None
        req_es = _clean_text(row[spanish_col]) if spanish_col else None
        if req_es is None:
            if allow_untranslated:
                req_es = req_original
            else:
                raise ValueError(
                    f"{path}: falta traducción al español en la fila {idx}. "
                    "El archivo SecReq debe incluir Requirement_ES/Requisito/Traduccion."
                )

        original_label = _clean_text(row[original_label_col]) if original_label_col else None
        final_label = _normalize_general_label(row[final_col]) if final_col else None

        if ambiguous_class_col:
            ambiguous_value = _clean_text(row[ambiguous_class_col])
            as_general = _normalize_general_label(ambiguous_value)
            if as_general in GENERAL_LABELS:
                final_label = final_label or as_general
            elif original_label is None:
                original_label = ambiguous_value

        # Solo `sec` permite una asignación provisional directa. `nonsec` NO implica usabilidad.
        if final_label is None and _norm_name(original_label or "") == "sec":
            final_label = "Security"

        security, usability = _label_to_binary(final_label)
        status = "ready" if final_label in GENERAL_LABELS else "pending_review"
        rows.append(
            {
                "Source": "SecReq",
                "Project": _clean_text(row[project_col]) if project_col else path.stem,
                "Source_ID": _clean_text(row[id_col]) if id_col else str(idx),
                "Requirement_original": req_original,
                "Requirement_ES": req_es,
                "Original_label": original_label,
                "Classification": final_label,
                "Security": security,
                "Usability": usability,
                "Label_status": status,
            }
        )
    return rows


def _duplicate_key(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dataset(
    promise_paths: list[Path],
    secreq_paths: list[Path],
    output_path: Path,
    allow_untranslated: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in promise_paths:
        rows.extend(_promise_rows(path, allow_untranslated=allow_untranslated))
    for path in secreq_paths:
        rows.extend(_secreq_rows(path, allow_untranslated=allow_untranslated))

    if not rows:
        raise ValueError("No se generaron registros. Revise rutas, columnas y etiquetas.")

    out = pd.DataFrame(rows)
    out.insert(0, "ID", np.arange(1, len(out) + 1, dtype=int))
    out["Duplicate_key"] = out["Requirement_ES"].map(_duplicate_key)
    counts = out["Duplicate_key"].value_counts(dropna=False)
    out["Is_exact_duplicate"] = out["Duplicate_key"].map(counts).fillna(0).gt(1)
    out.drop(columns=["Duplicate_key"], inplace=True)

    out["Security"] = out["Security"].astype("Int64")
    out["Usability"] = out["Usability"].astype("Int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # Reporte compacto junto al CSV para auditoría.
    report_path = output_path.with_suffix(".summary.txt")
    ready = out[out["Label_status"] == "ready"]
    pending = out[out["Label_status"] != "ready"]
    summary = [
        f"Output: {output_path}",
        f"Total registros: {len(out)}",
        f"Listos para modelado: {len(ready)}",
        f"Pendientes de revisión: {len(pending)}",
        f"Duplicados exactos marcados: {int(out['Is_exact_duplicate'].sum())}",
        "",
        "Distribución de Classification (solo listos):",
        ready["Classification"].value_counts(dropna=False).to_string(),
        "",
        "Distribución por fuente:",
        out["Source"].value_counts(dropna=False).to_string(),
    ]
    report_path.write_text("\n".join(summary), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promise", nargs="+", default=[], help="CSV(s) PROMISE traducidos")
    parser.add_argument("--secreq", nargs="+", default=[], help="CSV(s) SecReq traducidos")
    parser.add_argument("--output", default="dataset_consolidado.csv", help="CSV de salida")
    parser.add_argument(
        "--allow-untranslated",
        action="store_true",
        help="Permite usar el texto original cuando falta español. Solo para depuración/pruebas.",
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
        raise ValueError("Debe indicar al menos un archivo con --promise o --secreq")

    df = build_dataset(
        promise_paths=promise_paths,
        secreq_paths=secreq_paths,
        output_path=Path(args.output),
        allow_untranslated=args.allow_untranslated,
    )
    print(df.head().to_string(index=False))
    print(f"\nDataset guardado en: {args.output}")
    print(f"Resumen guardado en: {Path(args.output).with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
