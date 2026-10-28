# Clasificación multietiqueta de requisitos de seguridad

Cuatro modelos (CNN, BiLSTM, BETO, RoBERTa-BNE) entrenados **por separado** sobre tres datasets
(PROMISE_exp, SecReq, DOSSPRE), en dos condiciones: **C0** (`c0_real`, solo requisitos reales) y
**C1** (`c1_augmented`, reales + sintéticos generados con LLM, solo en entrenamiento). Cada modelo
predice `Security` y cuatro subcategorías de seguridad de ISO/IEC 25010 (`Confidentiality`,
`Integrity`, `Authenticity`, `Resistance`). `Non_repudiation` y `Accountability` se descartaron el
25-10-2026 por falta de datos.

## Estructura

```
inputs/                    datos de entrada: lo que se preprocesa y lo que se entrena
  original/     archivos fuente, sin modificar
  interim/      glosario, traducciones, preanotaciones y aumento (augmentation/)
  training/     promise.csv, secreq.csv, dosspre.csv  <- los datos reales que leen los modelos
outputs/                   todo lo que generan el entrenamiento y los reportes
  output_raw/   datos crudos (CSV/JSON/estudios de Optuna) para procesar por programa
    splits/                         folds externos e internos y distribución de etiquetas
    <condición>/<dataset>/<modelo>/ métricas, trials, predicciones, curvas internas, matriz de confusión (CSV)
    <condición>/model_comparison.csv, scott_knott_esd.csv y los análisis entre modelos (ver abajo)
    synthetic_diversity.csv         diversidad de los sintéticos (G9)
  reports/      reportes legibles
    <condición>/results_<condición>.xlsx
    <condición>/confusion_matrices/<dataset>_<modelo>.png
  logs/         salida de cada corrida
  checkpoints/  modelos guardados con --save-checkpoints (no se versiona)
src/
  config.py     rutas, etiquetas, condiciones, hiperparámetros y espacios de búsqueda de Optuna
  data.py       carga validada, CV anidada y unión de sintéticos por fold
  metrics.py    métricas multietiqueta y umbrales por etiqueta
  models.py     arquitecturas y codificación del texto
  training.py   rutina de entrenamiento
  tuning.py     estudio de Optuna sobre los folds internos
  pipeline.py   corrida completa de un trío (condición, dataset, modelo)
  reporting.py  matrices de confusión, tabla comparativa y Excel de resultados (sin reentrenar)
  significance.py  Scott-Knott ESD no paramétrico, Q de Cochran, McNemar exacto y Holm
  analysis.py   pruebas entre modelos, confusión entre subcategorías, errores, mejor modelo,
                Transformers vs. DL y diversidad de sintéticos (sin reentrenar)
scripts/
  prepare_datasets.py      original/ -> training/
  verify_subcategories.py  cobertura de las subcategorías
  extend_trials.py         archiva una etapa terminada para extender su búsqueda de Optuna
  build_report.py          regenera matrices de confusión, comparaciones, análisis y el Excel
  compare_conditions.py    pruebas estadísticas C0 vs. C1 (requiere las dos condiciones)
  glossary/                flujo de PLAN_TRADUCCION_GLOSARIO.md (P0-P8, E2)
  augmentation/            flujo de PLAN_AUMENTO_DATOS.md (G1, G3, G4, G5-G7; g9_diversity.py)
train.py                   una corrida: --condition C --dataset X --model Y
run_all_training.py        las corridas de una condición
tests/                     pruebas rápidas del protocolo (no entrenan)
docs/
  planes/                  planes consolidados de cada parte del flujo
  contexto_sesion/         contextos de sesión, reportes de corridas, resultados y revisiones (provisionales)
  reporte_tecnico/         espejo markdown del artículo en Google Docs (gdoc/) y su sincronización (_sync/)
  normas_apa_ulima/        guía APA 7 de la Universidad de Lima en markdown
  papers/                  papers de referencia (leer primero papers/markdowns/)
old/                       código y corridas anteriores (ver old/run_2026-10-19_sonnet_deepl_holdout/)
```

## Uso

```powershell
pip install -r requirements.txt
python scripts/prepare_datasets.py       # reconstruye inputs/training/
python scripts/verify_subcategories.py
python -m unittest discover -s tests -v
python run_all_training.py --condition c1_augmented --models cnn roberta_bne
python train.py --condition c0_real --dataset secreq --model beto
python scripts/build_report.py           # reportes de corridas ya terminadas, sin reentrenar
```

Una corrida interrumpida se retoma desde el último fold externo terminado. `--save-checkpoints`
guarda el modelo de cada fold externo en `outputs/checkpoints/` (no se guarda por defecto).

## Datasets de entrenamiento

| Dataset | Filas | Origen de `Security` | Origen de las subcategorías |
|---|---:|---|---|
| PROMISE_exp | 460 | `SE` vs. resto de RNF | preanotación de `claude-sonnet-5-5` (`llm_provisional`) |
| SecReq | 440 | `sec` / `nonsec` | preanotación de `claude-sonnet-5-5` (`llm_provisional`) |
| DOSSPRE | 1262 | `SR` / `NSR` de *2.0Binary*, salvo Availability y Maintainability | clase de *1.0Original* mapeada a ISO/IEC 25010 |

- Se eliminan duplicados exactos y, en PROMISE, las filas sintéticas del archivo balanceado.
- `Spanish_promise_exp_nfr_test_recovered.csv` se reconstruyó desde el antiguo dataset unido: el
  archivo original ya no estaba en el proyecto.
- SecReq y DOSSPRE están en inglés; se tradujeron con DeepL + glosario (corrida provisional) porque
  BETO y RoBERTa-BNE son modelos en español. Mapeo de clases de DOSSPRE: Confidentiality y
  Authorisation -> Confidentiality; Integrity -> Integrity; Authentication e Identification ->
  Authenticity; Immunity, IntrusionD y Survivability -> Resistance. Auditing queda con
  `Security=1` y sin subcategoría (Accountability se descartó). Availability y Maintainability son
  `SR` en DOSSPRE (Kadebu et al., 2023), pero en ISO/IEC 25010 pertenecen a Fiabilidad y
  Mantenibilidad: pasan a `Security=0`, igual que las clases A y MN de PROMISE.
- Las preanotaciones de PROMISE y SecReq son provisionales: no sustituyen la validación experta.

### Cobertura (positivos reales)

| Etiqueta | PROMISE_exp | SecReq | DOSSPRE |
|---|---:|---:|---:|
| Security | 79 | 177 | 406 |
| Confidentiality | 37 | 31 | 123 |
| Integrity | 14 | 60 | 33 |
| Authenticity | 18 | 70 | 92 |
| Resistance | 10 | 12 | 120 |

Una etiqueta con menos de 10 positivos queda fuera de los promedios (macro/weighted/micro-F1);
hoy las cinco son evaluables en los tres datasets.

### Sintéticos (C1)

`inputs/interim/augmentation/synthetic_<dataset>.csv`: 92 (PROMISE), 90 (SecReq) y 139 (DOSSPRE)
requisitos generados por `claude-sonnet-5-5`, traducidos con DeepL + glosario y verificados a
ciegas por 3 corridas del mismo modelo (corrida provisional, sin revisión experta). Cada uno guarda
su `Anchor_ID`: el requisito real del que se generó. Detalle en
[`docs/planes/PLAN_AUMENTO_DATOS.md`](docs/planes/PLAN_AUMENTO_DATOS.md).

## Protocolo (por dataset, idéntico para los cuatro modelos y las dos condiciones)

1. **Folds externos**: 5 folds con estratificación multietiqueta iterativa (semilla 42). Cada
   requisito real se evalúa exactamente una vez.
2. **Folds internos**: dentro del entrenamiento de cada fold externo, 3 folds para Optuna.
3. **Optuna** (TPE + poda por mediana): el objetivo es el macro-F1 medio de los folds internos con
   umbral 0.50. El trial 0 siempre es la configuración por defecto. Trials por fold externo
   (`N_TRIALS`): 25 para CNN y BiLSTM, 12 para BETO y RoBERTa-BNE. Se alcanzan en dos etapas
   (15/15/6/6 y luego `scripts/extend_trials.py`); la etapa anterior queda en `stage_<n>trials/`.
   Estado de C1: BETO y RoBERTa-BNE con 12 trials; CNN y BiLSTM todavía con 15.
4. **Umbrales por etiqueta** (0.40–0.60): con las predicciones out-of-fold internas del ganador.
5. **Modelo del fold externo**: se entrena con todo el entrenamiento externo durante la media de
   las mejores épocas internas y predice una vez el fold externo de prueba.
6. **Métricas**: media ± desviación de los 5 folds externos y métricas de las predicciones
   externas agrupadas.
7. **C1**: un sintético entra en un entrenamiento (interno o externo) solo si su `Anchor_ID` está
   en él. Validación y prueba contienen solo requisitos reales
   (`tests/test_protocol.py::test_synthetic_rows_never_leak_into_evaluation`).

### Hiperparámetros

Los valores por defecto son los `INITIAL_CONFIG` de los notebooks originales y los rangos de
búsqueda son sus `SEARCH_SPACE` (ambos en `src/config.py`).

| Modelo | Por defecto | Búsqueda |
|---|---|---|
| CNN | emb 64, filtros 100, kernel 3, batch 32, 50 épocas, dropout 0.2, lr 1e-3 | emb {64,100,128}, filtros {100,128}, kernel {3,4,5}, batch {16,32}, épocas {50,60}, dropout {0.2–0.5}, lr 1e-4–1e-2 |
| BiLSTM | emb 100, hidden 64, 1 capa, batch 32, 30 épocas, dropout 0.2, lr 1e-3 | emb {64,100,128}, hidden {32,64,128}, capas {1,2}, batch {16,32,50}, épocas {20,30,50}, dropout {0.2,0.3,0.5}, lr 5e-4–2e-3 |
| BETO / RoBERTa-BNE | lr 2e-5, batch 16, 5 épocas, dropout 0.1, weight decay 0.01, AdamW | lr 1e-5–5e-5, batch {8,16}, épocas {5,10,15,20}, dropout 0.1–0.5, weight decay 1e-8–0.3, optimizador {AdamW, Adam} |

Diferencias respecto de los notebooks originales:

- `pos_weight` (BCE ponderada por `min(sqrt(N_neg/N_pos), 10)`, calculada solo con el entrenamiento)
  y `warmup_ratio` (solo transformers) se añadieron a la búsqueda; sus valores por defecto (`False`
  y `0.0`) reproducen el comportamiento original.
- `threshold` ya no es un hiperparámetro: se usa 0.50 durante la búsqueda y luego los umbrales OOF.
- `max_length` queda fijo en 128: solo 2 de los 2178 requisitos lo superan.
- En BETO y RoBERTa-BNE un fold se detiene tras 3 épocas sin mejora de macro-F1 ni de loss de
  validación; `epochs` actúa como máximo.
- RoBERTa-BNE se carga desde el mirror `Buzzeitor/roberta-base-bne`, porque el repositorio oficial
  ya no publica los pesos.

## Archivos crudos de cada corrida (`outputs/output_raw/<condición>/<dataset>/<modelo>/`)

| Archivo | Contenido |
|---|---|
| `summary.json` | resumen completo: folds externos, métricas agrupadas, configuración por fold |
| `outer_fold_metrics.csv` | métricas de cada fold externo |
| `pooled_outer_test_predictions.csv`, `pooled_metrics_by_label.csv` | predicciones externas agrupadas y F1 por etiqueta |
| `pooled_confusion_by_label.csv` | matriz de confusión 2×2 de cada etiqueta (VP, FP, FN, VN) sobre las predicciones agrupadas |
| `stage_<n>trials/` | resultado de una etapa anterior de Optuna, archivado por `extend_trials.py` |
| `outer_<k>/outer_result.json` | configuración ganadora, umbrales y prueba del fold externo k |
| `outer_<k>/optuna_trials.csv`, `optuna_study.db`, `trials/` | estudio de Optuna del fold externo (reanudable) |
| `outer_<k>/inner_cv_history.csv`, `inner_cv_curves.png` | curvas internas del ganador |
| `outer_<k>/outer_test_predictions.csv` | predicciones del fold externo de prueba |

## Resultados de una condición

`train.py` actualiza estos archivos al terminar cada corrida; `scripts/build_report.py` los
regenera desde los archivos crudos guardados.

| Archivo | Contenido |
|---|---|
| `outputs/output_raw/<condición>/model_comparison.csv` | una fila por (dataset, modelo): métricas generales y F1 por etiqueta |
| `outputs/output_raw/<condición>/scott_knott_esd.csv` | rango y grupo de Scott-Knott ESD de cada modelo por dataset |
| `outputs/output_raw/<condición>/model_tests_cochran_q.csv`, `model_tests_mcnemar.csv` | Q de Cochran por (dataset, etiqueta) y McNemar exacto por pares con Holm |
| `outputs/output_raw/<condición>/subcategory_confusion.csv` | confusión entre subcategorías de cada (dataset, modelo), formato largo |
| `outputs/output_raw/<condición>/misclassified.csv` | requisitos con algún error, con las predicciones de cada modelo y el texto |
| `outputs/output_raw/<condición>/best_model.csv` | mejor modelo de cada dataset y motivo |
| `outputs/output_raw/<condición>/family_comparison*.csv` | Transformers frente a Deep Learning, por dataset y por etiqueta |
| `outputs/output_raw/synthetic_diversity.csv` | similitudes LaBSE de los sintéticos (más `synthetic_diversity_labse.csv` si se corrió `g9_diversity.py`) |
| `outputs/reports/<condición>/results_<condición>.xlsx` | libro formateado para leer los resultados (ver abajo) |
| `outputs/reports/<condición>/confusion_matrices/` | figura de las matrices de confusión de cada (dataset, modelo) |

Hojas de `results_<condición>.xlsx` (en negrita, el mejor modelo de cada dataset; escala de color
de 0 a 1 en las métricas):

| Hoja | Contenido |
|---|---|
| Resumen general | métricas generales agrupadas y media ± desviación de los 5 folds externos |
| F1 por clase | F1 de cada etiqueta y macro-F1, una fila por (dataset, modelo) |
| Métricas por clase | precisión, recall, F1, soporte y VP/FP/FN/VN de cada etiqueta (con filtros) |
| Matrices de confusión | matriz 2×2 de cada etiqueta, un bloque por (dataset, modelo) |
| Folds externos | métricas generales de cada fold externo |
| Scott-Knott ESD | rango y grupo de cada modelo por dataset (ver abajo) |
| Cochran Q y McNemar | prueba ómnibus por (dataset, etiqueta) y comparaciones por pares |
| Confusión subcategorías | matriz subcategoría real × predicha de cada (dataset, modelo) |
| Errores para revisión | requisitos que algún modelo falla, ordenados por cuántos modelos fallan |
| Mejor modelo | modelo elegido en cada dataset con la regla de METODOLOGIA §4.3.2 |
| Transformers vs DL | comparación de familias por dataset y por etiqueta |
| Diversidad sintéticos | solo en C1: similitudes LaBSE de los sintéticos aceptados |

### Scott-Knott ESD

Implementación no paramétrica descrita por Binkhonain y Alfayez (2025, §3.7), sobre el macro-F1
de los 5 folds externos de cada modelo: los modelos se ordenan por mediana, se busca el corte
con mayor estadístico de Kruskal-Wallis y se divide si es significativo (α = 0,05), de forma
recursiva; después se fusionan los grupos adyacentes con δ de Cliff despreciable. El paper no da
el umbral; se usa |δ| < 0,147, el que aplica el paquete de R ScottKnottESD. El rango 1 es el grupo
superior; los modelos con el mismo rango no se distinguen. Con 5 observaciones por modelo la
prueba solo separa diferencias grandes.

### Pruebas y análisis entre modelos

- **Q de Cochran + McNemar:** sobre las predicciones externas agrupadas, por dataset y etiqueta.
  Si la Q de Cochran de los 4 modelos es significativa, se comparan los 6 pares con McNemar
  exacto y p ajustado por Holm (Limaylla-Lunarejo et al., 2026; Shafikuzzaman et al., 2025).
- **Confusión entre subcategorías:** la celda (i, j) cuenta los requisitos con la subcategoría
  i que el modelo omitió y a los que asignó j, que no tienen; «Ninguna» recoge omisiones sin
  sustituta y subcategorías asignadas a requisitos que no tienen ninguna.
- **Mejor modelo:** los modelos del grupo superior de Scott-Knott ESD no difieren en macro-F1;
  entre ellos se elige el de mayor recall de Security agrupado (Petrillo et al., 2025: ante F1
  iguales, el recall distingue al modelo que omite menos requisitos de seguridad).
- **Transformers vs. DL:** media de los dos modelos de cada familia; δ de Cliff y Mann-Whitney
  sobre los 10 macro-F1 de folds externos de cada familia.
- **Diversidad de sintéticos (G9):** `python scripts/augmentation/g9_diversity.py` calcula con
  LaBSE la similitud media sintético-sintético, real-real y sintético-real (necesita
  `transformers`); el Excel la incluye al regenerarse.
