"""Create an expert-review queue for rare labels and freeze the current test IDs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

LABELS = ["Security", "Confidentiality", "Integrity", "Non_repudiation", "Accountability", "Authenticity", "Resistance"]

# Suggestions are hypotheses for expert review, never automatic ground-truth edits.
SUGGESTIONS = {
    372: (0, 0, "Control de acceso a solicitudes de disputa; 'disputa' es el dominio, no evidencia de una acción."),
    373: (0, 0, "Autenticación/login y autorización; no expresa prueba frente a repudio."),
    374: (0, 0, "Autenticación/login; 'Sistema de disputas' es el nombre del sistema."),
    375: (0, 1, "Informe de cambios revisado con fines de auditoría: Accountability clara; Non-repudiation no explícita."),
    458: (0, 0, "Nivel de acceso para iniciar disputas; control de acceso, no no-repudio."),
    459: (0, 0, "Restricción de acceso directo a datos; Confidentiality probable, no no-repudio."),
    475: (1, 0, "Prueba de que ocurrió una carga/descarga para evitar repudio: caso inequívoco de Non-repudiation."),
    837: (0, 0, "Mecanismo de disputa/reembolso por error MAC; requiere experto, no menciona evidencia probatoria."),
    838: (0, 1, "Repositorio central de errores: trazabilidad/Accountability; 'disputa' es contexto del proceso."),
    187: (0, 1, "Auditoría, conservación de mensajes y seguimiento de transacciones: Accountability clara."),
    274: (0, 1, "Log atribuible a IP de origen de la transacción: Accountability clara."),
    313: (0, 0, "Identificación única usada para login: Authenticity clara; atribución/auditoría no explícita."),
    347: (0, 0, "'Usuarios registrados' activó la heurística; el requisito general no exige trazabilidad."),
    376: (0, 1, "Registro de altas y cambios de acceso en informe: Accountability clara."),
    423: (0, 0, "'Agentes registrados' describe autorización; no registro de acciones."),
    694: (0, 1, "Asocia sesión, dispositivo y usuario invocador: atribución inequívoca de acciones."),
    695: (0, 0, "Timestamp/secuencia del mensaje puede ser Integrity/Resistance; atribución no explícita."),
    696: (0, 0, "Verificación anti-replay: Resistance/Integrity; no Accountability por sí sola."),
    723: (0, 1, "ID único y contador persistente de transacciones permiten trazabilidad."),
}

root = Path(".")
data = pd.read_csv(root / "dataset_seguridad_provisional.csv", encoding="utf-8-sig")
manifest = pd.read_csv(root / "artifacts/security_split_manifest_seed42.csv", encoding="utf-8-sig")
merged = data.merge(manifest[["ID", "split", "text_sha256"]], on="ID", validate="one_to_one")
review = merged[merged["Non_repudiation"].eq(1) | merged["Accountability"].eq(1)].copy()
review["Suggested_Non_repudiation"] = review["ID"].map(lambda value: SUGGESTIONS[int(value)][0])
review["Suggested_Accountability"] = review["ID"].map(lambda value: SUGGESTIONS[int(value)][1])
review["Audit_rationale"] = review["ID"].map(lambda value: SUGGESTIONS[int(value)][2])
review["Expert_decision_Non_repudiation"] = pd.NA
review["Expert_decision_Accountability"] = pd.NA
review["Expert_comments"] = ""
review["Review_status"] = "pending_expert_review"
columns = [
    "ID", "split", "Requirement_ES", *LABELS,
    "Suggested_Non_repudiation", "Suggested_Accountability", "Audit_rationale",
    "Expert_decision_Non_repudiation", "Expert_decision_Accountability",
    "Expert_comments", "Review_status", "Subcategory_label_status", "Prelabel_evidence",
]
review[columns].sort_values("ID").to_csv(
    root / "dataset_seguridad_v2_rare_labels_expert_review.csv", index=False, encoding="utf-8-sig"
)

frozen = manifest[manifest["split"].eq("test")].sort_values("ID").copy()
frozen.to_csv(root / "artifacts/security_test_frozen_seed42.csv", index=False, encoding="utf-8-sig")
digest = hashlib.sha256((root / "artifacts/security_test_frozen_seed42.csv").read_bytes()).hexdigest()
(root / "artifacts/security_test_frozen_seed42.sha256").write_text(
    f"{digest}  security_test_frozen_seed42.csv\n", encoding="ascii"
)
print("Review rows:", len(review), "| frozen test rows:", len(frozen), "| sha256:", digest)
