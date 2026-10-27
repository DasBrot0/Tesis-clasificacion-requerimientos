# Corrida provisional sin validaciones (18-10-2026)

**Todo lo de este documento es PROVISIONAL: no hubo validación humana ni de expertos.** Ejecutado por Claude Code (subagentes `claude-sonnet-5-5` para las tareas LLM).

## 1. Qué se corrió y qué se omitió

| Paso | Estado | Notas |
|---|---|---|
| P3 | usado tal cual (SON) | sin pasos 3 a 6 |
| P5 | hecho (provisional) | 661 términos, 23 lotes de 29, 1 subagente nuevo por lote |
| P6 | hecho | DeepL Free, glosario, `quality_optimized` |
| P7 | omitido | |
| P8 | hecho sin chrF++ | exactitud terminológica (exacta y por lema, spaCy `es_core_news_lg`), CometKiwi, LaBSE |
| P9 | omitido | queda la traducción de DeepL |
| P10 | hecho | `Translation_source=deepl+glossary_provisional` (SecReq, DOSSPRE) / `promise_exp_translated` |
| E2 | hecho | guía v1 provisional + 3 corridas SON, voto mayoritario |
| E8 | hecho | `llm_provisional` para PROMISE y SecReq |
| Entrenamiento | **lanzado, en curso** | ver sección 7 |

Resultados anteriores respaldados (movidos, no borrados) en `old/run_2026-10_regex_opusmt/` (`artifacts/`, `checkpoints/`, `datasets/training/`, y `prepare_datasets_legacy.py`). Las pruebas de humo están en `old/smoke_test_2026-10-18/`.

## 2. Glosario y DeepL

- **Entradas finales: 643** (`datasets/interim/glossary/glossary_en_es_provisional.tsv`; `decided_by=SON_provisional`, primera propuesta de SON).
- Construcción (`scripts/glossary/p5_proposals.py build`): clave EN = forma superficial más frecuente en el corpus (se prefiere la singular si tiene al menos la mitad de la frecuencia de la más común); NE y siglas con el mismo texto en ES; sin claves duplicadas, sin ES vacíos ni espacios sobrantes. Si la forma del corpus es plural, se pluraliza el ES con una regla simple (no se agregaron entradas plurales extra).
- **Limpieza mínima añadida por CC (desvío, sin criterio experto):** la primera versión (657 entradas, `p5/glossary_v1_sin_limpiar.tsv`) tenía entradas dañinas, p. ej. `secure -> SECURE` (que convertía todo *secure* en "SECURE") y fragmentos (`logical channel(s`, `O interface(s`, `client re`, `Zimbabwe Anti`, `theIntrusionD`, `ES`, `manage`, `contract -> contrato inteligente`, `template -> plantilla biométrica`, etc.). Se descartaron 14 (lista en `p5/build_log.txt`) y se retradujo todo el corpus.
- **ID del glosario DeepL: `3607abad-ebe0-4517-aeb7-8aa1e807285d`.** El plan Free admite un único glosario, así que no se pudo crear uno nuevo: se actualizó este mismo ID (PUT `/v3/glossaries/{id}/dictionaries`), sin borrar nada. Hash v2 de entradas: `5bf02f573b58` (v1: `cad26e05261a`, ya sustituido). Registro en `datasets/interim/translations/deepl_glossary_provisional.json`.
- Traducción: `translations/deepl_provisional.jsonl`, 1 727 textos únicos (DOSSPRE 1 283 + SecReq 444), lotes de 50, respuestas crudas en `translations/deepl_raw/`. La primera tanda (glosario v1) quedó en `translations/v1_glosario_sin_limpiar/`.
- **Caracteres DeepL:** uso total 437 882 de 1 000 000 (≈ 123 de una prueba mía; dos pasadas completas de ≈ 218 800 y reintentos). Restan ≈ 562 000 este mes.
- Incidencia: para algunos textos DeepL devolvió el inglés sin traducir al aplicar el glosario (2 con "NGN network"/"SAG system", p. ej. *The SAG system shall perform well…*). Se reintentó **sin glosario** (campo `note` y `es_glossary_attempt` en el JSONL, `glossary_id=null`). Mi filtro de detección fue demasiado amplio: 8 textos quedaron retraducidos sin glosario (4 de ellos no lo necesitaban: Zimbabwe Anti-Corruption, "These merchant acquirers", dos de "Internally to the CPN"). Pendiente de revisar.

### Antes / después (DOSSPRE; antes = `old/dosspre_es_translation.csv`, opus-mt)
| EN | Antes | Después |
|---|---|---|
| …detect SQL Injection and XSS vulnerabilities… | "vulnerabilidades SQL Inyection y XSS" | "vulnerabilidades de inyección SQL y XSS" |
| …push warning script to the rogue node | "nodo rogue" | "nodo malicioso" |
| …the firewall should be able to withstand… | "cortafuegos" | "cortafuegos" |
| …classify it as spam or ham | "spam o jamón" | "spam o correo legítimo" |

(*firewall* ya salía "cortafuegos" en DOSSPRE; en SecReq antes quedaba sin traducir.)

## 3. Control de calidad (P8, `translations/qc_signals_provisional.csv`, `qc_summary_provisional.md`)

- CometKiwi (1 727 textos): media 0,864; mediana 0,875; p10 0,819; p25 0,852; p75 0,887; mín 0,536; máx 0,903. 4 textos < 0,70 y 96 < 0,80.
- Exactitud terminológica: por lema == 1 en el **85,9 %** de los textos (por coincidencia exacta 82,3 %); 1 108 textos tienen al menos un término del glosario; media 0,929 en ellos.
- LaBSE (con `transformers`, `pooler_output` normalizado, venv de CometKiwi): media 0,883; < 0,80 en 72 textos (4,2 %).
- Peores 20 por CometKiwi y por términos faltantes: `qc_summary_provisional.md`. Los peores por términos son frases con "MACing key signature", "secure data exchange protocols" y "Load File Data Block Hash": DeepL flexiona o reordena el término y la comparación por lema no lo reconoce, o el glosario tiene entradas solapadas.
- Pendiente: los textos "SAG system" y "NGN network" conservan una `term_acc` baja porque el glosario no se aplicó en el reintento.

## 4. Etiquetas de subcategorías (E2)

- Guía: `notas_sesion/guia_anotacion_subcategorias_v1_provisional.md` (marcada como redactada por CC, pendiente de expertos). Prompt: `datasets/interim/labels/prompt_v1.md`; hash prompt+guía `4b65e1e3aee9`. Entradas: 256 (PROMISE 79, SecReq 177), en inglés.
- 3 corridas × 9 lotes de 29, un subagente nuevo por lote y corrida, validados con JSON schema. Salidas congeladas en `labels/runs/run_{1,2,3}`, `son_prelabels_run{1,2,3}.csv`, y `son_prelabels.csv` (voto mayoritario). No se controlan `temperature` ni `seed`. **Salvedad:** los subagentes son contextos separados pero comparten prompt y guía; algunos subagentes (p. ej. run_2/b04, run_3/b01) escribieron su JSONL mediante un script auxiliar con anotaciones escritas a mano en lugar de la herramienta Write (mismo contenido de juicio, sin leer otras corridas, según su informe).
- Acuerdo entre corridas: unanimidad en las 6 etiquetas simultáneamente 87,5 %; por etiqueta 96,5–99,2 %. κ medio por pares: Confidentiality 0,96; Integrity 0,96; Authenticity 0,95; Resistance 0,95; Accountability 0,71; Non_repudiation 0,66 (`labels/agreement_between_runs.json`).
- Sin ninguna subcategoría: 6 de 79 en PROMISE y 39 de 177 en SecReq.

### Positivos por etiqueta (nuevo vs. antiguo regex/opus-mt)
| Etiqueta | PROMISE nuevo/viejo | SecReq nuevo/viejo | DOSSPRE nuevo/viejo |
|---|---|---|---|
| Filas | 460 / 460 | 440 / 439 | 1 262 / 1 279 |
| Security | 79 / 79 | 177 / 177 | 501 / 503 |
| Confidentiality | 37 / 41 | 31 / 31 | 123 / 124 |
| Integrity | 14 / 5 | 60 / 47 | 33 / 33 |
| Non_repudiation | 0 / 6 | 2 / 3 | 0 / 0 |
| Accountability | 4 / 7 | 4 / 5 | 38 / 38 |
| Authenticity | 18 / 23 | 70 / 65 | 92 / 92 |
| Resistance | 10 / 9 | 12 / 13 | 120 / 121 |

- Con `MIN_POSITIVES_PER_LABEL=10`: **no evaluables** Non_repudiation (PROMISE 0, SecReq 2, DOSSPRE 0) y Accountability (PROMISE 4, SecReq 4). `verify_subcategories.py` termina con código 1 por esa razón (esperable).
- Cambio frente al regex (comparado sobre el español nuevo, `--report-regex-agreement`, `labels/regex_vs_son_agreement.txt`): **PROMISE 46 de 474 etiquetas (9,7 %)**, **SecReq 73 de 1 062 (6,9 %)**. DOSSPRE no cambia de mapeo (`DOSSPRE_CLASS_MAP`).
- Las filas de DOSSPRE bajan (1 279 → 1 262) por deduplicar sobre el inglés y luego por traducciones idénticas (inglés distinto, mismo español: 20 filas fusionadas con unión de etiquetas); SecReq 444 → 440 por lo mismo (3 de traducción idéntica + 1).

## 5. Cambios de código

- `scripts/prepare_datasets.py` (original en `old/run_2026-10_regex_opusmt/prepare_datasets_legacy.py`): SecReq desde `source_en.csv` + DeepL; DOSSPRE con DeepL (ya no opus-mt ni `old/`); PROMISE conserva su español y toma las subcategorías de SON; deduplicación sobre el inglés, más una segunda pasada sobre el español (el cargador `src/data.py` exige textos únicos); nueva columna `Translation_source`; `Subcategory_label_status` = `llm_provisional` / `structural_zero` / `class_mapped` (en DOSSPRE se conservan también `no_iso25010_equivalent` y `class_mapped_merged_duplicates`). El regex queda como `legacy_regex_labels`, sin uso salvo `--report-regex-agreement`.
- Nuevos: `scripts/glossary/p5_proposals.py`, `p6_deepl.py`, `p8_qc.py`, `p8_neural_signals.py` (venv aislado), `e2_prelabel.py`; `scripts/run_provisional_training.py`.
- Pruebas: `python -m pytest tests -q` → 6 pasan. (`python -m pytest -q` en la raíz también recoge `extras/legacy_cv/`, que falla por un archivo de datos inexistente; es código heredado ajeno a estos cambios.)
- No se instaló nada en el Python principal; no se usó el SDK `deepl` (REST con `requests`).

## 6. Entrenamiento: pruebas de humo
- `promise cnn --n-trials 1` (47 s) y `promise roberta_bne --n-trials 1` (2,6 min) corrieron de punta a punta con los datos nuevos. RoBERTa aprende (macro-F1 CV 0,27, test 0,29; todos los folds > 0). Se movieron a `old/smoke_test_2026-10-18/` y `artifacts/` y `checkpoints/` se reiniciaron vacíos.

## 7. Entrenamiento completo (en curso)
- Lanzado oculto con `scripts/run_provisional_training.py` (PID 12708 al lanzar; sobrevive al cierre de sesión), 12 pares en secuencia (promise cnn/bilstm/beto/roberta_bne; secreq; dosspre), continúa si uno falla. Cada par: `python train.py --dataset X --model Y` (Optuna con N_TRIALS de `src/config.py`: CNN y BiLSTM 30, BETO y RoBERTa 12).
- Log: `logs/provisional_training_20261018_223200.log` (marcas `=== dataset modelo: inicio/fin código=…`). Monitoreo: `grep "===" logs/provisional_training_*.log` y `tail` del log; procesos: `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`.
- Salidas: `artifacts/<dataset>/<modelo>/` (`summary.json`, `optuna_trials.csv`, `test_metrics_by_label.csv`, …), `artifacts/model_comparison.csv` (comparación global, se actualiza al terminar cada par), `checkpoints/<dataset>/<modelo>/final`.
- Al informar: `promise cnn` terminó (código 0, 6,5 min); `promise bilstm` en curso. Duración total esperada: muchas horas.
- Nota: `datasets/training/*.csv` se regeneró dos veces durante la corrida (el CSV de PROMISE es byte a byte idéntico, md5 `db7f8530…`); SecReq y DOSSPRE ya tenían su versión final antes de que sus pares empezaran.

## 8. Limitaciones / pendientes
- Glosario SON sin verificar con fuentes; limpieza mínima de CC no experta. Etiquetas SON sin κ frente a humanos. Corridas de SON no controladas en `temperature`/`seed`.
- Reintentos sin glosario (8 textos) a revisar; `term_acc` penaliza SAG/NGN.
