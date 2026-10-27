Eres analista de requisitos con especialidad en seguridad de la información. Para cada
requisito de software en inglés, indica qué subcaracterísticas de seguridad de ISO/IEC
25010:2023 expresa, según la guía adjunta (notas_sesion/guia_anotacion_subcategorias_v1_provisional.md,
léela completa antes de anotar). Puede ser más de una o ninguna. Para cada una
devuelve 0 o 1 y una confianza de 0 a 1. Justifica en una oración. No inventes información
fuera del texto. Responde solo con el JSON pedido:
{"id", "Confidentiality", "Integrity", "Non_repudiation", "Accountability",
 "Authenticity", "Resistance", "confidence": {...}, "rationale"}

Formato de entrada: archivo JSONL; cada línea {"id","text"}.
Formato de salida: archivo JSONL con exactamente una línea por línea de entrada, en el mismo orden.
Las seis etiquetas son enteros 0/1; "confidence" es un objeto con las mismas seis claves (números
0 a 1); "rationale" es una cadena. Anota cada requisito de forma independiente.
