# Plan de etiquetado de subcategorías de seguridad (ISO/IEC 25010:2023)

Versión 1 · 18-10-2026. Complementa a [`PLAN_TRADUCCION_GLOSARIO.md`](PLAN_TRADUCCION_GLOSARIO.md).

> **A quién va dirigido.** A los autores de la tesis, para ejecutar el plan, y a un LLM que
> redactará la metodología. Para el LLM redactor:
>
> - Cita **solo** las fuentes de la sección "Referencias" (2022 en adelante, revisadas por pares).
> - Lo marcado **[Decisión propia]** no tiene respaldo bibliográfico: preséntalo como decisión de
>   diseño de los autores, con su razón. Lo marcado **[NV]** no se pudo verificar: no lo afirmes.
> - **Usabilidad queda fuera de alcance.** El borrador del artículo todavía la incluye (14
>   subcategorías); hay que quitarla.
> - Las etiquetas de este plan son el **ground truth** con el que se entrenan y evalúan los
>   modelos. Por eso las deciden expertos y no los autores.

---

## 1. Qué se etiqueta

- **Etiquetas:** seis subcaracterísticas de seguridad de ISO/IEC 25010:2023, en formato
  **multietiqueta** (un requisito puede tener varias): Confidentiality, Integrity, Non_repudiation,
  Accountability, Authenticity y Resistance.
- **Solo se etiquetan los requisitos de seguridad.** La etiqueta `Security` viene de los datasets
  originales y no se cambia: PROMISE (`SE`), SecReq (`sec`) y DOSSPRE (`SR`, anotado por 3
  expertos; Kadebu et al., 2023). Los requisitos no de seguridad tienen las seis subcategorías en 0.
- **Se etiqueta sobre el texto en inglés.** Así las etiquetas no dependen de la traducción, el
  etiquetado puede hacerse en paralelo con el plan de traducción y una traducción nueva no cambia
  ninguna etiqueta. **[Decisión propia]**
- **Las expresiones regulares dejan de generar etiquetas.** Las que hay hoy en
  `scripts/prepare_datasets.py` (`PATTERNS`) solo se conservan para compararlas con las etiquetas
  finales.

| Dataset | Requisitos de seguridad | Origen de las subcategorías en este plan |
|---|---|---|
| PROMISE_exp | 79 | Anotación de expertos (pasos E4 y E5) |
| SecReq | 177 (textos únicos en inglés) | Anotación de expertos (pasos E4 y E5) |
| DOSSPRE | 503 | Conversión de la clase original, auditada por un experto (paso E6) |

---

## 2. Actores

| Código | Quién | Qué hace en este plan |
|---|---|---|
| **A1** | Autor 1 (Franco) | Redacta la guía con A2, prepara planillas y muestras, ejecuta la preanotación, modera las reuniones de consenso, calcula los acuerdos. **No decide etiquetas.** |
| **A2** | Autora 2 (coautora) | Redacta la guía con A1, construye y justifica la tabla de conversión de DOSSPRE, verifica las planillas. **No decide etiquetas.** |
| **EXP1** | Profesor(a) de ciberseguridad que asesora tesis del área (confirmado como probable) | Experto principal: piloto, muestra a ciegas, revisión del resto de PROMISE y SecReq, consenso. |
| **EXP2** | Segundo(a) profesor(a) de ciberseguridad | Segundo experto: piloto, muestra a ciegas, consenso y auditoría de DOSSPRE. |
| **SON** | `claude-sonnet-5-5`, por la **suscripción de Claude**, como subagentes de Claude Code (sin API) | Preanota las subcategorías. **Nunca decide.** |
| **CC** | Claude Code | Scripts de muestreo, planillas, cálculo de κ e integración, bajo la supervisión de A1. |

- **No se usa ningún LLM local** (GPU de 4 GB).
- **Criterio de experto** (se reporta el perfil de cada uno): al menos 3 años de experiencia
  profesional o docente en seguridad de la información, o una certificación del área.
  **[Decisión propia]**, siguiendo la práctica de describir el perfil de los anotadores (Kadebu
  et al., 2023; Rejithkumar y Anish, 2025).
- **Si solo se consigue un experto:** EXP1 hace también las tareas de EXP2. A1 hace de segundo
  anotador en la muestra a ciegas, solo para medir el acuerdo; las etiquetas finales las decide
  EXP1, como en Wang et al. (2026), donde un experto adjudica lo que anotaron otros. Se declara como
  limitación.

---

## 3. Pasos

### E1 · Guía de anotación

- **Quién:** A1 y A2 redactan la versión 1; EXP1 la revisa antes del piloto.
- **Contenido:**
  1. La definición de cada subcaracterística según ISO/IEC 25010:2023:
     - Confidentiality: los datos solo son accesibles para quienes están autorizados.
     - Integrity: se impide el acceso o la modificación no autorizados de datos o programas.
     - Non_repudiation: se puede demostrar que acciones o eventos ocurrieron, de modo que no
       puedan negarse después.
     - Accountability: las acciones de una entidad se pueden rastrear de forma única hasta ella.
     - Authenticity: se puede demostrar que un sujeto o recurso es quien dice ser.
     - Resistance: el sistema sigue operando mientras está bajo ataque de un actor malicioso.
  2. Criterios de inclusión y exclusión para cada una, y la opción "ninguna aplica".
  3. **Reglas para casos de frontera**, que se completan en el piloto. Ejemplos a resolver:
     bloqueo tras varios intentos fallidos (¿Resistance, Authenticity o ambas?); registro de
     transacciones (¿Accountability o Non_repudiation?); control de acceso por roles
     (¿Confidentiality o Integrity?); firmas y MAC (¿Integrity, Authenticity o Non_repudiation?).
  4. **Regla contra palabras gatillo:** una palabra como *secure*, *protected*, *safe*, *password*
     o *log* no basta para marcar una subcategoría; se marca por lo que el requisito exige. Por
     ejemplo, "the system shall securely display the menu" no expresa ninguna subcaracterística por
     decir *securely*.
  5. La diferencia entre "ninguna aplica" (es de seguridad, pero no encaja en ninguna
     subcaracterística) y "no se puede decidir" (UNCLEAR, ver E2).
  6. Ejemplos resueltos tomados del propio corpus.
  7. Una casilla para marcar, sin cambiarlo, un requisito que el experto no considere de seguridad.
- **Respaldo:** Groen et al. (2025), guía con definiciones de ISO/IEC 25010 y ejemplos; Bagies
  (2024), reglas explícitas para las clases ambiguas; Rigouts Terryn (2026), el LLM recibe la misma
  guía que los anotadores humanos; IREB (2026), AI4RE Prompt Guide "Tag Requirements": los LLM
  confunden categorías ante palabras como *quick* o *intuitive*, de donde sale la regla del punto 4
  (documentación profesional, no un paper revisado por pares).
- **Salida:** `notas_sesion/guia_anotacion_subcategorias.md`, que se publica como apéndice.

### E2 · Preanotación con Sonnet

- **Quién:** SON, orquestado por CC.
- **Paso a paso:**
  1. Entrada: el texto en inglés de los 256 requisitos de seguridad de PROMISE y SecReq, y de la
     muestra de DOSSPRE del paso E6.
  2. Prompt (versión 2) = la guía de E1 (definiciones, reglas de frontera y ejemplos) más esta
     instrucción:
     ```text
     Eres analista de requisitos con especialidad en seguridad de la información. Para cada
     requisito de software en inglés, indica qué subcaracterísticas de seguridad de ISO/IEC
     25010:2023 expresa, según la guía adjunta. Puede ser más de una o ninguna. Para cada una
     devuelve 0 o 1 y una confianza de 0 a 1. No marques una subcaracterística solo porque aparezca
     una palabra como "secure", "protected" o "password": marca lo que el requisito exige.
     Si el texto no permite decidir si una subcaracterística aplica, inclúyela en "unclear" en vez
     de adivinar. Justifica en una oración. No inventes información fuera del texto. Responde solo
     con el JSON pedido:
     {"id", "Confidentiality", "Integrity", "Non_repudiation", "Accountability",
      "Authenticity", "Resistance", "confidence": {...}, "unclear": [...], "rationale"}
     ```
     La versión 1 (sin `unclear` ni la regla de palabras gatillo) es la que se usó en la corrida
     provisional: `datasets/interim/labels/prompt_v1.md`.
  3. Lotes de 30, un subagente nuevo por lote, **3 corridas independientes** y voto mayoritario por
     etiqueta. Se reporta el acuerdo entre corridas.
  4. Registro igual que en P3 del plan de traducción (modelo, versión de Claude Code, fecha UTC,
     hash del prompt, archivos). Las salidas se congelan.
- **Para qué sirve:** acelerar la revisión de EXP1 (paso E5) y estratificar las muestras. No es una
  etiqueta final. Un requisito con alguna etiqueta en `unclear` en la mayoría de las corridas pasa
  a E5 **sin preanotación**: EXP1 lo anota a ciegas.
- **Diferencias con la plantilla "Tag Requirements" de IREB** **[Decisión propia]**: IREB pide
  elegir el objetivo principal y aquí se permite más de una etiqueta, porque la tarea es
  multietiqueta; IREB marca el requisito completo como UNCLEAR y aquí se marca por
  subcaracterística, para no perder las etiquetas que sí están claras.
- **Respaldo:** Ahmed et al. (2025): el acuerdo entre modelos indica si la tarea es apta para un
  LLM, y la confianza del modelo ayuda a elegir qué ítems pueden delegarse; Wellhausen et al.
  (2026): el LLM como evaluador adicional, comparado con el κ entre humanos; IREB (2026), AI4RE
  Prompt Guide "Tag Requirements": categorías definidas por el usuario, justificación obligatoria y
  una salida UNCLEAR cuando no se puede decidir.
- **Salida:** `datasets/interim/labels/son_prelabels.csv`.

### E3 · Ronda piloto

- **Quién:** EXP1 y EXP2 anotan; A1 modera la discusión y actualiza la guía.
- **Paso a paso:**
  1. 30 requisitos de PROMISE y SecReq, distintos de los de E4.
  2. EXP1 y EXP2 los anotan por separado, sin ver la preanotación.
  3. Se discuten las diferencias y se ajusta la guía (versión 2). Estos 30 no entran al κ.
- **Respaldo:** Wellhausen et al. (2026): ronda de precodificación, guía revisada con ejemplos y
  segunda ronda.

### E4 · Muestra a ciegas para medir el acuerdo

- **Quién:** EXP1 y EXP2 anotan por separado; A1 calcula el acuerdo y modera el consenso.
- **Paso a paso:**
  1. Muestra de unos 75 requisitos de PROMISE y SecReq, estratificada por dataset. Incluye todos
     los que SON preanotó con una etiqueta rara (Non_repudiation, Accountability), para que su κ se
     pueda calcular. **[Decisión propia]** el tamaño.
  2. EXP1 y EXP2 anotan a ciegas: sin la preanotación y sin ver las respuestas del otro.
  3. Se reportan:
     - el acuerdo multietiqueta según Marchal et al. (2022);
     - el κ de Cohen por etiqueta;
     - el porcentaje de acuerdo.
  4. Los desacuerdos se resuelven en una reunión de consenso entre EXP1 y EXP2, con la guía. El
     consenso es la etiqueta final de esos 75.
  5. Si una etiqueta queda con κ < 0,4, se revisa su definición en la guía y EXP1 reanota esa
     etiqueta en los 256. **[Decisión propia]**
- **Respaldo:**
  - Rejithkumar y Anish (2025): κ multietiqueta sobre muestra de dos anotadores y reuniones de
    conciliación.
  - Marchal et al. (2022): el κ estándar no sirve para anotación multietiqueta; proponen métricas
    adecuadas.
  - Groen et al. (2025): desacuerdos conciliados.
  - Limaylla-Lunarejo et al. (2023): el acuerdo por subcategoría de ISO/IEC 25010 suele ser
    moderado (κ de Fleiss de 0,48, frente a 0,78 por categoría). **[NV]** Confirmar en el PDF.

### E5 · Revisión del resto de PROMISE y SecReq

- **Quién:** EXP1.
- **Paso a paso:**
  1. Se compara la preanotación de SON con el consenso de E4, etiqueta por etiqueta.
  2. Los requisitos con `unclear` (E2) los anota EXP1 a ciegas, siempre.
  3. **Si el κ de SON frente al consenso es ≥ 0,6** en una etiqueta, EXP1 revisa el resto (unos 180)
     con esa preanotación a la vista y la acepta o corrige. Si no, EXP1 anota esa etiqueta a ciegas.
     **[Decisión propia]** el umbral.
  4. Se registra el porcentaje de etiquetas que EXP1 cambió respecto de la preanotación.
- **Riesgo que se declara:** sesgo de anclaje al ver la preanotación. Se acota con el paso 2 y con
  el porcentaje de cambios.
- **Respaldo:** Ahmed et al. (2025): delegar parte de la anotación en un LLM no cambia
  significativamente el acuerdo cuando el LLM concuerda con los humanos tanto como ellos entre sí.
  No hay un trabajo de ingeniería de requisitos que valide este paso para subcategorías de
  seguridad: se declara como limitación.

### E6 · DOSSPRE: conversión auditada

- **Quién:** A2 construye y justifica la tabla; EXP2 audita; EXP1 y EXP2 deciden las correcciones.
- **Paso a paso:**
  1. A2 justifica cada fila de `DOSSPRE_CLASS_MAP` con la definición de la clase en Kadebu et al.
     (2023) y la de ISO/IEC 25010:2023, en una tabla que va al artículo:

     | Clase DOSSPRE | Subcaracterística | A justificar |
     |---|---|---|
     | confidentiality, authorisation | Confidentiality | *authorisation* es la más discutible |
     | integrity | Integrity | |
     | auditing | Accountability | |
     | authentication, identification | Authenticity | |
     | immunity, intrusion detection, detection, survivability | Resistance | revisar *survivability* |
     | availability, maintainability | ninguna (fiabilidad y mantenibilidad en ISO/IEC 25010) | |

  2. Muestra de unos 100 requisitos de seguridad de DOSSPRE, estratificada por clase original.
  3. EXP2 los anota a ciegas, con la guía y sin ver la clase original.
  4. Se mide, clase por clase, el acuerdo entre la conversión y EXP2. Si una clase tiene acuerdo
     bajo, EXP1 y EXP2 deciden una regla corregida y se documenta.
  5. Se reporta cuántos requisitos de la muestra tienen una segunda subcategoría: es la limitación
     de que DOSSPRE tenga una sola clase por requisito.
  6. Non_repudiation queda fuera en DOSSPRE: no hay ninguna clase que corresponda.
  7. **Duplicados del original con etiquetas contradictorias.** DOSSPRE trae requisitos repetidos
     con diferencias mínimas en inglés (puntuación, artículos, erratas): 21 pares en DOSSPRE y 4 en
     SecReq. Se fusionan en una sola fila, porque con el mismo texto en español serían una fuga de
     datos entre entrenamiento y evaluación. Hoy se fusionan uniendo las etiquetas. En 5 pares de
     DOSSPRE las copias tienen etiquetas distintas, y EXP2 decide la etiqueta de cada uno (se
     registra la decisión):

     | Requisito (inglés) | Copia 1 | Copia 2 |
     |---|---|---|
     | "The system shall be able to open the door when the pin is correct" | SR, Authorisation | NSR, Performance |
     | "The system shall be able to do random check on connection" | NSR, Performance | SR, Availability |
     | "The system shall allow a user to sign a document on behalf of the group using their secret key" | SR, Authorisation | SR, Integrity |
     | "The system shall allow anomalies to be curbed after detection" | SR, IntrusionD | SR, Immunity |
     | "The system has clearly defined links and buttons that show the various functionalities…" | NSR, Functionality | NSR, Usability |

     Los dos primeros cambian la etiqueta `Security`. Es la única excepción a la regla de no
     modificar `Security`, porque el propio dataset se contradice. **[Decisión propia]**
  8. En la corrida final, la deduplicación sobre el inglés normaliza también la puntuación y los
     artículos, para que estos pares se detecten ahí y se reporten como duplicados del original.
- **Respaldo:** Kadebu et al. (2023): las clases originales las anotaron 3 expertos; Bagies (2024):
  conversión de las clases de DOSSPRE a confidencialidad, integridad y disponibilidad con
  definiciones y reglas para los casos ambiguos; Limaylla-Lunarejo et al. (2026): tabla de
  equivalencia de PROMISE a ISO/IEC 25010.

### E7 · Etiquetas con pocos positivos

- Una etiqueta con menos de 10 positivos en un dataset se reporta como **no evaluable** y sale del
  promedio macro (ya implementado: `MIN_POSITIVES_PER_LABEL = 10` en `src/config.py`).
- Junto a cada F1 se reporta el número de positivos.
- **[Decisión propia]**

### E8 · Integración

- **Quién:** CC, bajo la supervisión de A1.
- **Paso a paso:**
  1. Las etiquetas finales se guardan en `datasets/interim/labels/labels_en.csv` por `Source_ID`.
  2. `scripts/prepare_datasets.py` las toma de ahí en vez de las expresiones regulares, y las hereda
     cada traducción.
  3. `Subcategory_label_status` toma uno de estos valores: `expert_consensus` (E4),
     `expert_reviewed` (E5), `class_mapped_audited` (E6) o `llm_provisional` (corrida provisional).
  4. Opcional: se reporta el acuerdo entre las expresiones regulares antiguas y las etiquetas
     finales, como evidencia de por qué se reemplazaron.

### E9 · Qué se reporta

- El perfil de EXP1 y EXP2.
- Los acuerdos de E4 (multietiqueta, por etiqueta y porcentaje), el κ de SON frente al consenso y
  el porcentaje de cambios de E5.
- El acuerdo por clase de E6.
- Los positivos finales por etiqueta y dataset.
- La guía, como apéndice.

---

## 4. Carga de trabajo

| Quién | Qué | Tiempo aprox. |
|---|---|---|
| EXP1 | 30 piloto + 75 a ciegas + unos 180 de revisión + consenso | 3 horas |
| EXP2 | 30 piloto + 75 a ciegas + 100 de DOSSPRE + consenso | 2,5 a 3 horas |
| A1 y A2 | Guía, tabla de DOSSPRE, planillas, preanotación, κ, reuniones | — |

Conviene pedirles a los expertos un bloque concreto (por ejemplo, dos sesiones de 1,5 horas) y
entregarles las planillas listas.

---

## 5. Corrida provisional sin validaciones (18-10-2026)

Para tener resultados preliminares mientras se consigue a los expertos:

- **PROMISE y SecReq:** subcategorías = voto mayoritario de las 3 corridas de SON (paso E2), con
  la guía en su versión 1 redactada por CC. Estado `llm_provisional`.
- **DOSSPRE:** la conversión actual, sin auditar.
- **Sus resultados son provisionales y no son los finales de la tesis.** Sirven para el avance con
  el asesor y para estimar cuánto cambian las etiquetas respecto de las expresiones regulares.

---

## 6. Amenazas a la validez

| Amenaza | Mitigación |
|---|---|
| Fuera de la muestra de E4, cada requisito lo revisa un solo experto | Se reporta el κ de la muestra; se declara |
| Sesgo de anclaje en E5 | Umbral de κ de SON y porcentaje de cambios reportado |
| DOSSPRE tiene una sola clase por requisito | Auditoría de E6 y conteo de segundas subcategorías |
| Etiquetas raras | No evaluables con menos de 10 positivos |
| La etiqueta `Security` original podría ser discutible | No se cambia; se cuenta cuántos requisitos marcan los expertos como no de seguridad. Única excepción: los 2 duplicados de DOSSPRE que se contradicen en `Security` (E6, punto 7) |
| Se etiqueta en inglés y se entrena en español | La traducción sigue un plan con glosario y control de calidad; la etiqueta describe el contenido del requisito, no su redacción |

---

## 7. Correcciones pendientes en el borrador del artículo

Según la revisión del 18-10-2026 de `D:\ULima\2026-1\Propuesta de investigacion\markdowns\`:

- Quitar usabilidad (las 8 subcategorías y la etiqueta general).
- Reemplazar "dos expertos etiquetarán la totalidad del dataset" por el procedimiento de este plan.
- Nombrar la métrica de acuerdo (Marchal et al., 2022) y describir la conversión de DOSSPRE.
- Incluir DOSSPRE en la sección 4.1.
- **Petrillo et al. (2025) no reetiqueta**: usa SecReq tal cual. No citarlo como respaldo del
  etiquetado.
- **Rejithkumar y Anish (2025)** no usó dos expertos sobre todo el dataset: 5 analistas que se
  repartieron los requisitos y conciliaron desacuerdos.
- **Kadebu et al. (2023)** tiene 10 clases de seguridad (no "once") y no reporta el valor del κ.

---

## 8. Referencias (solo las que respaldan decisiones)

- Bagies, T. (2024). Classifying software security requirements into confidentiality, integrity,
  and availability using machine learning approaches. *PeerJ Computer Science, 10*, e2554.
  https://doi.org/10.7717/peerj-cs.2554
- Kadebu, P., Sikka, S., Tyagi, R., & Chiurunge, P. (2023). A classification approach for software
  requirements towards maintainable security. *Scientific African, 19*, e01496.
  https://doi.org/10.1016/j.sciaf.2022.e01496
- Rejithkumar, G., & Anish, P. (2025). NICE: Non-functional requirements identification,
  classification, and explanation using small language models. *ICSE-SEIP 2025*, 284–295.
  https://doi.org/10.1109/ICSE-SEIP66354.2025.00031
- Groen, E. C., Dalpiaz, F., van Vliet, M., Winter, B., Doerr, J., & Brinkkemper, S. (2025).
  Classification of quality characteristics in online user feedback using linguistic analysis,
  crowdsourcing and LLMs. *Journal of Systems and Software, 230*, 112533.
  https://doi.org/10.1016/j.jss.2025.112533
- Wellhausen, C., Reinhardt, L., & Schneider, K. (2026). User reviews as a source for usability
  requirements: A precursor study on using large language models. *REFSQ 2026 Workshops*,
  CEUR-WS. **[Completar]** volumen de CEUR.
- Wang, W., Peng, J., Xiao, H., Hua, Y., Zhou, Y., Li, Z., & Wang, X. (2026). Machine learning,
  deep learning, or large language models: An empirical study on multi-label requirements
  classification. *Empirical Software Engineering, 31*, 149.
  https://doi.org/10.1007/s10664-026-10879-3
- Marchal, M., Scholman, M., Yung, F., & Demberg, V. (2022). Establishing annotation quality in
  multi-label annotations. *COLING 2022*, 3659–3668. https://aclanthology.org/2022.coling-1.322/
- Ahmed, T., Devanbu, P., Treude, C., & Pradel, M. (2025). Can LLMs replace manual annotation of
  software engineering artifacts? *MSR 2025*. https://doi.org/10.1109/MSR66628.2025.00086
- Rigouts Terryn, A. (2026). Large language models are good term extractors: A systematic
  evaluation. *LREC 2026*, 7631–7643. https://aclanthology.org/2026.lrec-1.605/
- Limaylla-Lunarejo, M.-I., Condori-Fernández, N., & Rodríguez Luaces, M. (2023). Towards a FAIR
  dataset for non-functional requirements. *ACM SAC '23*, 1414–1421.
  https://doi.org/10.1145/3555776.3578611. **[NV]** los valores de κ.
- Limaylla-Lunarejo, M.-I., Condori-Fernández, N., Rodríguez Luaces, M., & Karras, O. (2026).
  *Empirical Software Engineering, 31*, 6. https://doi.org/10.1007/s10664-025-10736-9
- ISO/IEC. (2023). *ISO/IEC 25010:2023 Product quality model*.
- International Requirements Engineering Board. (2026). *Prompt Guide: Tag Requirements*. En
  *AI4RE Prompt Guide*. IREB e.V. Recuperado el 18 de octubre de 2026 de
  https://ireb.atlassian.net/wiki/spaces/airebpromptguide/pages/2430369836/Prompt+Guide+Tag+Requirements
  (página creada el 13-02-2026; versión 3 del 12-06-2026). Documentación profesional, no revisada
  por pares.
