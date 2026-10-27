# Contexto de la sesión del 17–18 de octubre de 2026

Resumen de todo lo conversado y hecho en esta sesión, para retomar el trabajo sin perder
contexto. El plan detallado y vigente está en [`PLAN_TRADUCCION_GLOSARIO.md`](PLAN_TRADUCCION_GLOSARIO.md) (en esta carpeta desde el 18-10-2026);
este archivo explica cómo se llegó a él, qué se decidió y qué queda pendiente.

---

## 1. Proyecto y foco

- **Tesis:** clasificación multietiqueta de requisitos de **seguridad** en español. Se predicen
  `Security` y las seis subcategorías de ISO/IEC 25010: Confidentiality, Integrity,
  Non_repudiation, Accountability, Authenticity y Resistance.
- **Modelos:** CNN, BiLSTM, BETO y RoBERTa-BNE.
- **Datasets:** PROMISE_exp, SecReq y DOSSPRE.
- Ver `README.md` del proyecto.
- **La traducción EN→ES y el glosario son preparación de datos, no el objeto de estudio.**

## 2. Restricciones y preferencias del autor

- **Solo fuentes de 2022 en adelante.** La documentación técnica vigente de herramientas se cita
  como tal.
- **Variante de destino: español de España**, por cercanía a RoBERTa-BNE, que se preentrenó con el
  corpus de la Biblioteca Nacional de España.
- **Verbos modales (*shall*, *must*…) fuera de alcance.** No son relevantes para el foco de
  seguridad.
- **LLM: solo `claude-sonnet-5-5`**, ejecutado como **subagentes de Claude Code** y no por API.
  - Se evaluaron como alternativas gpt-oss-120b en Groq (gratuito o plan Developer), la ejecución
    local en una ThinkStation P5 (RTX 6000 Ada de 48 GB y 256 GB de RAM), modelos más livianos
    (gpt-oss-20b) y la API de Anthropic (Haiku o Sonnet).
  - **Decisión final: solo Sonnet 5.5.** Motivos:
    - P3 solo prioriza; la decisión final es humana en P5.
    - La validez la da el κ entre el LLM y el humano.
    - La falta de un modelo abierto se declara como limitación (Sallou et al., 2024, recomiendan
      comparar con modelos abiertos, pero no es obligatorio).
- **Modelo abierto de traducción (SalamandraTA) descartado.** No hay GPU suficiente en la laptop y
  la traducción no es el foco.
- **Segundo revisor humano (P5 y P9): pendiente.**

## 3. Investigación realizada (con subagentes)

### 3.1 Paper antecedente: Limaylla-Lunarejo et al. (2026)

*Empirical Software Engineering* 31:6, doi:10.1007/s10664-025-10736-9. Salió online el
2 de noviembre de 2025.

- **Cómo tradujeron PROMISE_exp al español:**
  1. Tradujeron con Google Translate y con DeepL desde sus páginas web.
  2. Midieron la similitud coseno entre las dos traducciones con word2vec SBWCE
     (`SBW-vectors-300-min5.bin`, mediante gensim).
  3. Las traducciones con similitud < 0,97 (unas ¾) las revisaron expertos a mano.
  4. En las de similitud ≥ 0,97, la primera autora eligió la más "natural".
- **No usaron glosario** ni control terminológico, y no reportan el acuerdo entre revisores.
  Ellos mismos reconocen sesgo y baja replicabilidad (§8).
- **Debilidades del método:**
  - Compara dos traducciones automáticas entre sí, nunca contra el original.
  - Promediar word2vec ignora el orden y la negación.
  - El umbral de 0,97 no está calibrado.
- El glosario de esta tesis se presenta como mejora sobre ese antecedente. No se encontró ningún
  trabajo que traduzca datasets de requisitos con glosario, así que puede plantearse como aporte.

### 3.2 IREB (ireb.org)

- **"Prompt Guide: Tag Requirements"** (AI4RE Prompt Guide, wiki de IREB). Es una plantilla de
  clasificación con categorías definidas por el usuario: si encajan varias, elige el objetivo
  principal; si no se puede decidir, UNCLEAR; pide justificación. Advierte que los LLM confunden
  funcional y no funcional ante palabras como "quick" o "intuitive".
- **AI4RE Syllabus & Study Guide v1.0.0 (2026):** describe RTF, CRISPE, Chain-of-Thought,
  zero-shot, few-shot y prompts basados en plantilla.
- **Glosario CPRE en español v2.2.1:** términos oficiales (requisito de calidad, restricción,
  fiabilidad, tolerancia a faltas; Security = Seguridad, Safety = Protección). Solo los términos
  están traducidos; las definiciones siguen en inglés. **Casi no contiene el vocabulario técnico de
  los requisitos** (browser, encrypt, password, backup…).
- **Glosario de Glinz:** sus definiciones de requisito funcional, no funcional, de calidad y
  restricción coinciden palabra por palabra con el system prompt del paper antecedente.
- **CPRE Syllabus 3.2 (2024):** "Use terms consistently, as defined in the glossary".
- **IREB no tiene guías para traducir requisitos.**
- Copias en texto de los PDF de IREB: carpeta `scratchpad/pdfs/` de la sesión (temporal).

### 3.3 Fuentes de 2022 en adelante para el método del glosario

Las referencias completas y su estado de verificación están en la sección de referencias del plan.

| Paso | Fuentes |
|---|---|
| Patrones nominales con spaCy | Zhang et al. (2022, RE); Palacios Palacios et al. (2025, RANLP, en español); Marciniak et al. (2023, EACL) |
| Umbral de frecuencia y hápax | Marciniak et al. (2023); Palacios Palacios et al. (2025). **El valor ≥ 2 es decisión propia** |
| C-value | Marciniak et al. (2023); Zhang (2022, *Scientific Reports*); Banerjee et al. (2024); Tran et al. (2026, *ACM Computing Surveys*) |
| Especificidad frente a un corpus general | Kyröläinen & Laippala (2023); Palacios Palacios et al. (2025). wordfreq (Speer, 2022) se cita solo como herramienta |
| LLM en extracción de términos | Rigouts Terryn (2026, LREC); Tran et al. (2025, revista *Terminology*) |
| Traducción con terminología | Semenov et al. (2023, 2025, WMT; el WMT25 evaluó EN→ES en el dominio de TI); Moslem et al. (2023 ×2); Bogoychev & Chen (2023) |
| Control de calidad | CometKiwi (Rei et al., 2022, 2023); Freitag et al. (2023, 2024); Amrhein & Sennrich (2022); LaBSE (Feng et al., 2022) |
| Alineamiento de palabras con LaBSE (P4) | Wang et al. (2022, Findings of EMNLP), **[NV]** |
| Riesgos de los LLM en la investigación | Sallou et al. (2024); Ahmed et al. (2025, MSR); Shah et al. (2025, NLP4RE); Ouyang et al. (2024); Sainz et al. (2023) |
| Terminología normalizada | ISO 704:2022; ISO/IEC 25010:2023; INCIBE (2024); IATE |

- Las fuentes anteriores a 2022 (Justeson & Katz 1995, Frantzi et al. 2000, Dunning 1993,
  Arora et al. 2017, etc.) **no se usan** por la restricción del autor.

### 3.4 Revisión del código y los datos del proyecto

- `scripts/prepare_datasets.py:39` busca `SecReq_es.csv` en `datasets/original/`, pero **ese
  archivo ya no está ahí**; solo queda la copia en `old/`. **El script falla si se ejecuta hoy.**
- SecReq se tradujo fuera del código y no se sabe con qué herramienta. Perdió el texto en inglés y
  la especificación de origen.
- La deduplicación actual se hace sobre el texto **en español**, lo que fusiona requisitos distintos
  en inglés (SecReq 444 → 439; DOSSPRE 1 283 → 1 279).
- Las subcategorías de SecReq y PROMISE se preanotan con expresiones regulares sobre el español, así
  que una traducción nueva cambia las etiquetas.
- `.env` contiene `DEEPL_AUTH_KEY`, `GROQ_API_KEY` y `GROQ_MODEL`. Ningún código del pipeline
  principal los usa todavía.

### 3.5 RoBERTa-BNE: diagnóstico (no se ha corregido nada)

- **La fuente es válida.** El mirror `Buzzeitor/roberta-base-bne` (commit
  `ef2fa9120bd771fd420893123a53ac9bfd8f0687`, sha256 del `.bin` `2bb8e5dc…7b75`) reproduce
  exactamente las puntuaciones de *fill-mask* de la ficha oficial. Otros dos mirrors independientes
  tienen los mismos archivos. El repositorio oficial `PlanTL-GOB-ES` ya no tiene los pesos y no hay
  copia en BSC-LT.
- **Causa del bajo rendimiento: el dropout.** `src/models.py:156-157` pone `hidden_dropout_prob` y
  `attention_probs_dropout_prob` = `config["dropout"]` (0,1–0,5). RoBERTa-BNE se preentrenó con
  dropout 0,0.
  - Con 0,1, macro-F1 = 0,00. Con 0,0, sube a 0,50–0,77 en pruebas cortas de un fold.
  - En BETO no afecta, porque se preentrenó con 0,1.
- **Problemas secundarios:**
  - No hay warmup por defecto (`src/training.py:56-57`) y 5 épocas son pocas.
  - `src/tuning.py:237`: `TPESampler(seed=SEED)` repite trials al reanudar un estudio.
  - Hay trials en estado RUNNING en `artifacts/promise/roberta_bne/optuna_study.db`.
- **Recomendaciones:**
  1. Aplicar el dropout solo a `classifier_dropout`.
  2. Usar épocas en {10, 15, 20} y un warmup de 0,06–0,1.
  3. Borrar `artifacts/promise/roberta_bne/` y relanzar.
  4. Corregir la reanudación de Optuna.
  5. Fijar `revision` del mirror.
  6. Citar MarIA: Gutiérrez-Fandiño et al. (2022), *Procesamiento del Lenguaje Natural* 68,
     doi:10.26342/2022-68-3.
  7. Corregir la conclusión equivocada de `old/AJUSTE_HIPERPARAMETROS_SEGURIDAD.md:93`.
- Los scripts de diagnóstico estaban en `scratchpad/roberta_check/` de la sesión (temporal).

## 4. Lo que se ejecutó: P0 a P4

Scripts en `scripts/glossary/`; salidas en `datasets/interim/glossary/`. El detalle y los números
están en la sección "Estado de ejecución" del plan.

| Paso | Script | Resultado principal |
|---|---|---|
| P0 | `p0_source.py` | `source_en.csv`: DOSSPRE 1 316 filas (1 283 únicas) y SecReq 510 (444 únicas). Se repararon las comillas corruptas de SecReq y se realineó la traducción antigua. |
| P1–P2 | `p1_p2_terms.py` | spaCy `en_core_web_trf`: 4 800 candidatos → 1 618 (frecuencia ≥ 2 y G² > 0; 126 siglas). Ranking por C-value, G² y dispersión. |
| P3 | `p3_llm_annotation.py` + `p3/prompt_v1.md` | Sonnet 5.5 como subagentes: 18 lotes de 90, 3 corridas, todo válido. Unanimidad: categoría 0,775; seguridad 0,950. Resultado: 496 Specific, 330 Common, 71 NE, 48 OOD, 673 NotTerm; 324 de seguridad. `human_sample.csv` (200) generado. |
| P4 (ahora diagnóstico preliminar, ver §6) | `p4_baseline_consistency.py` | Alineamiento con LaBSE (capa 6, argmax en ambas direcciones). El 31,5 % de los términos con 2 o más ocurrencias se tradujo de forma inconsistente; consistencia media 0,881. Ejemplos: *SQL injection* → "sql inyection"/"inyección sql"; *rogue* → "rogar"; *ham* → "jamón"; *firewall* sin traducir. |
| Lista para P5 | `p3b_shortlist.py` | `p5_shortlist.csv`: **661 términos** (Specific, de seguridad o NE). Antes eran 778; ver §6. |

- Dependencias instaladas: spaCy 3.8.16 (+ `en_core_web_trf` 3.8.0 y `es_core_news_lg`),
  wordfreq 3.1.1, python-dotenv, jsonschema y sacrebleu 2.6.0.
- `scripts/glossary/p3_groq_annotation.py` quedó escrito y probado con un lote, pero **no se usa**:
  se decidió quedarse solo con Sonnet. Se puede borrar o conservar como referencia.

## 5. Pendientes, en orden sugerido

1. **Anotar `datasets/interim/glossary/human_sample.csv`** (200 candidatos, sin mirar las etiquetas
   del LLM). Da el κ entre el LLM y el humano.
2. **Validar o reemplazar los 8 ejemplos *few-shot*** de `p3/prompt_v1.md`. Los redactó el
   asistente y el plan decía que los haría el autor.
3. **Decidir el alcance de P5:** los 661 términos o un subconjunto priorizado (por ejemplo, los 319
   de seguridad + los NE).
4. **P5:** las propuestas de equivalentes las hace Sonnet; el autor fija el glosario
   `glossary_en_es.tsv` con fuentes (prioridad: ISO/IEC 25010 → INCIBE → IREB → IATE; ver §7).
5. **Definir el segundo revisor** para P5 y P9.
6. **P6–P9:**
   - P6: DeepL con glosario (`target_lang=ES`).
   - P7: Sonnet con el glosario filtrado por requisito.
   - P8: exactitud terminológica, CometKiwi `wmt22-cometkiwi-da`, chrF++ entre motores y LaBSE.
   - P9: muestra calibrada de unos 150 requisitos, dos revisores y κ.
   - Faltan instalar `deepl`, `unbabel-comet` y `sentence-transformers`.
7. **P10:** actualizar `prepare_datasets.py`:
   - leer SecReq desde el inglés;
   - deduplicar sobre el inglés;
   - agregar `Translation_source`;
   - volver a ejecutar las preanotaciones y reportar cuántas cambian.
8. **Corregir RoBERTa-BNE** (sección 3.5) y relanzar sus entrenamientos.
9. ~~Verificar las referencias marcadas [NV]~~ Hecho (§7). Solo queda el fascículo de EMSE (se cita
   como 31:6, artículo 6).

## 6. Cambio posterior (18-10-2026): las traducciones previas salen del proceso

- **Decisión del autor:** la traducción anterior (DOSSPRE con opus-mt; SecReq con una herramienta
  desconocida) se hizo sin control terminológico y se considera metodológicamente incorrecta. No
  interviene en el proceso nuevo, que parte solo del inglés.
- **No se oculta:** en la tesis se reporta como *diagnóstico preliminar* que motiva el glosario
  (31,5 % de términos repetidos traducidos de forma inconsistente, *rogue* → "rogar", etc.). La
  comparación de antes y después es opcional y solo en DOSSPRE, que es reproducible.
- **Archivos movidos a `old/`:**
  - `old/dosspre_es_translation.csv` (antes en `datasets/interim/`). `old/SecReq_es.csv` ya estaba ahí.
  - `old/glossary_p4/`: `p4_baseline_consistency.py`, `term_baseline_renderings.csv`,
    `term_inconsistency_baseline.csv`, `p4_meta.json` y `p5_shortlist_con_p4.csv` (la lista de 778).
- **Cambios en el código:**
  - `p0_source.py` ya no agrega `ES_baseline`. Se verificó que `source_en.csv` es idéntico al
    anterior salvo esas columnas, así que P1–P3 siguen vigentes.
  - `p4b_shortlist.py` se renombró a `p3b_shortlist.py` y perdió el criterio D. La lista bajó de
    778 a 661: los 117 términos que entraban solo por D eran todos *Common* u *OOD*.
- La diferencia 324/319 en los términos de seguridad son 5 términos clasificados como *NotTerm*.
- El número P4 queda sin usar en el plan para no renumerar P5–P11.
- **Ojo:** si se ejecuta hoy `scripts/prepare_datasets.py`, no encuentra la traducción guardada y
  vuelve a traducir DOSSPRE con opus-mt (y sigue fallando en SecReq). No ejecutarlo hasta P10.

## 7. Cambios del 18-10-2026 (noche)

- **RoBERTa-BNE corregido:** el dropout ajustado va solo al clasificador (`src/models.py`); el
  mirror queda fijado por commit (`HF_REVISIONS` en `src/config.py`); la reanudación de Optuna ya
  no repite configuraciones (`src/tuning.py`). El estudio roto de PROMISE quedó en
  `old/artifacts_roberta_dropout/`. Afecta también a BETO: sus estudios hay que repetirlos.
- **Reentrenamiento tras la traducción nueva:** Optuna desde cero para los 4 modelos en SecReq y
  DOSSPRE, y para BETO y RoBERTa en PROMISE. CNN y BiLSTM en PROMISE se conservan si `promise.csv`
  queda idéntico (comparar el hash).
- **DeepL verificado con la clave del proyecto:** plan Free, 1 000 000 caracteres al mes; EN→ES
  con glosario y `quality_optimized` funcionan.
- **Plan actualizado:**
  - P3 con un segundo anotador (unos 50 términos).
  - P5 solo con fuentes abiertas y una regla para términos sin fuente.
  - P9 con sobremuestreo del cuartil bajo.
  - Se quitaron los preprints Baltes et al. (2025) y Ghazvininejad et al. (2023), reemplazados por
    Sallou et al. (2024), Ahmed et al. (2025) y Semenov et al. (2023).
- **CometKiwi verificado:** funciona en un entorno aparte (`C:/Users/Franco/.venvs/cometkiwi`, Python 3.12), en unos 4 minutos
  para el corpus en CPU. Separa errores graves y omisiones (0,57–0,59 frente a 0,84–0,88), pero
  apenas penaliza un error de término (0,816). Esto justifica medir la exactitud terminológica aparte.

## 8. Planes reorganizados y corrida provisional (18-10-2026)

- Los planes vigentes están en esta carpeta:
  - `PLAN_TRADUCCION_GLOSARIO.md` (versión 2): paso a paso, actores y solo las referencias que
    respaldan decisiones. La versión 1 está en `old/PLAN_TRADUCCION_GLOSARIO_v1.md`.
  - `PLAN_ETIQUETADO_SUBCATEGORIAS.md` (versión 1): las subcategorías las deciden expertos en
    ciberseguridad (EXP1 y EXP2), se etiqueta sobre el inglés y Sonnet solo preanota.
- Actores: A1 (Franco), A2 (coautora), EXP1 (profesor de ciberseguridad que asesora tesis del
  área, probable), EXP2 (otro profesor de ciberseguridad), Sonnet 5.5 por suscripción, DeepL Free
  y modelos locales no generativos. No se usa ningún LLM local.
- Usabilidad queda fuera de alcance.
- Hay un subagente ejecutando la corrida provisional sin validaciones (sección "Corrida
  provisional" de cada plan). Su informe irá a `CORRIDA_PROVISIONAL_2026-10-18.md`. Antes de
  regenerar los datos mueve los resultados anteriores a `old/run_2026-10_regex_opusmt/`.
