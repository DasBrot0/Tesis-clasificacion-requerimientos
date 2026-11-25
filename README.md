# Clasificación de requisitos de seguridad en español

Cinco modelos (CNN, BiLSTM, BETO, RoBERTa-BNE y DistilBETO) entrenados por separado sobre tres datasets
(PROMISE_exp, SecReq y DOSSPRE), en **dos tareas** (replanteo del 24-11-2026, pedido del asesor):

| Tarea | Datos | Salidas | Condiciones |
|---|---|---|---|
| 1. `binary` | todos los requisitos | 1: seguridad / no seguridad | C0 (solo reales) |
| 2. `subcategories` | solo los requisitos de seguridad | 4: Confidentiality, Integrity, Authenticity, Resistance (ISO/IEC 25010) | C0 y C1 (reales + sintéticos) |

Las dos tareas se combinan después como **cascada** (`src/cascade.py`): si el modelo binario dice
«seguridad», el modelo de subcategorías asigna las subcategorías; si no, ninguna. El modelo anterior,
que tenía 5 salidas y una regla de coherencia, se descartó; sus corridas están en
`old/run_2026-10-27_c1_pre_dedup/`.

## Uso

```powershell
pip install -r requirements.txt
python scripts/prepare_datasets.py          # inputs/original -> inputs/training
python scripts/verify_subcategories.py      # positivos por etiqueta
python -m unittest discover -s tests -v     # pruebas rápidas (no entrenan)

python run_all_training.py                  # todo: binary C0 + subcategories C0 y C1
python train.py --task binary --dataset secreq --model beto
python train.py --task subcategories --condition c1_augmented --dataset dosspre --model cnn

python scripts/build_report.py              # reportes y cascada sin reentrenar
python scripts/compare_conditions.py        # C0 vs. C1 en subcategorías
```

Una corrida interrumpida retoma desde el último fold externo terminado y, dentro de él, desde el
último trial de Optuna. `--save-checkpoints` guarda el modelo de cada fold externo.

## Cómo leer el código

El flujo de una corrida (`train.py` → `src/pipeline.py`) es:

```
prepare_datasets.py   fuentes -> <dataset>_binary.csv y <dataset>_subcategories.csv
src/data.py           carga, folds externos (compartidos por las dos tareas) e internos, sintéticos
src/pipeline.py       por cada fold externo:
  src/tuning.py         Optuna: cada trial entrena los 3 folds internos (src/training.py: fit)
  src/metrics.py        umbral por etiqueta con las predicciones fuera de fold del mejor trial
  src/training.py       modelo final con todo el entrenamiento externo -> predicción del fold de prueba
src/models.py         Dataset, collators (palabras / subpalabras) y las arquitecturas
src/config.py         todo lo ajustable: tareas, épocas, paciencia, hiperparámetros, búsqueda
```

`src/reporting.py`, `src/analysis.py`, `src/significance.py` y `src/cascade.py` solo leen lo que las
corridas guardaron (no entrenan).

## Estructura

```
inputs/
  original/       archivos fuente, sin modificar
  interim/        glosario, traducciones, etiquetas, duplicados manuales y aumento (augmentation/)
  training/       <dataset>_binary.csv, <dataset>_subcategories.csv  <- lo que leen los modelos
                  <dataset>.csv (tabla completa: base de los dos anteriores y de las planillas de validación)
outputs/
  output_raw/splits/                     folds externos (compartidos) e internos (por tarea)
  output_raw/<tarea>/<condición>/<dataset>/<modelo>/   archivos de cada corrida
  output_raw/<tarea>/<condición>/        comparaciones entre modelos
  output_raw/cascade/<condición>/        evaluación de extremo a extremo de la cascada
  reports/<tarea>/<condición>/           results_<tarea>_<condición>.xlsx y matrices de confusión
  logs/                                  <tarea>_<condición>_<dataset>_<modelo>.log
scripts/
  prepare_datasets.py, verify_subcategories.py, build_report.py, compare_conditions.py
  glossary/       flujo de PLAN_TRADUCCION_GLOSARIO.md (P0-P8, E2)
  augmentation/   flujo de PLAN_AUMENTO_DATOS.md (G1-G9)
old/              código y corridas anteriores
```

La documentación (planes, notas, reporte técnico y papers) está en el repo hermano
[Tesis-clasificacion-requerimientos-docs](https://github.com/DasBrot0/Tesis-clasificacion-requerimientos-docs),
en `../Tesis-clasificacion-requerimientos-docs/`.

## Datasets

| Dataset | Tarea 1 (todos) | Tarea 2 (seguridad) | Sin subcategoría | Origen de `Security` | Origen de las subcategorías |
|---|---:|---:|---:|---|---|
| PROMISE_exp | 456 | 77 | 9 | `SE` vs. el resto de RNF | voto de 3 corridas de `claude-sonnet-5-5` + revisión |
| SecReq | 433 | 177 | 44 | `sec` / `nonsec` | voto de 3 corridas de `claude-sonnet-5-5` + revisión |
| DOSSPRE | 1185 | 391 | 69 | `SR` / `NSR` de *2.0Binary*, salvo Availability y Maintainability | clase de *1.0Original* mapeada a ISO/IEC 25010 |

Positivos de la tarea 2:

| Subcategoría | PROMISE_exp | SecReq | DOSSPRE |
|---|---:|---:|---:|
| Confidentiality | 39 | 32 | 112 |
| Integrity | 14 | 66 | 61 |
| Authenticity | 18 | 74 | 121 |
| Resistance | 13 | 12 | 92 |

- Los requisitos de seguridad sin ninguna de las cuatro subcategorías (p. ej. Auditing de DOSSPRE)
  quedan en la tarea 2 con las cuatro etiquetas en 0.
- SecReq y DOSSPRE se tradujeron con DeepL + glosario; PROMISE_exp conserva el español del
  antecedente. Las correcciones revisadas de traducción y de etiquetas están en `inputs/interim/`.
- Duplicados: primero exactos (y se asignan los ID), luego casi duplicados (sin mayúsculas, tildes,
  puntuación ni, en inglés, artículos), sobre el inglés y sobre el español. Los ID que quedan no
  cambian, así que la numeración tiene huecos.
- Una etiqueta con menos de 10 positivos se predice, pero no entra en los promedios. Hoy todas son
  evaluables.

**Sintéticos (C1, solo tarea 2).** `inputs/interim/augmentation/synthetic_<dataset>.csv`: 92, 90 y 139
requisitos de seguridad generados por `claude-sonnet-5-5` (corrida provisional; el aumento completo
queda para Seminario II). Cada uno guarda su `Anchor_ID`, el requisito real del que se generó, y
solo entra en un entrenamiento si ese requisito está en él.

## Protocolo

1. **Folds externos:** 5, calculados una vez por dataset sobre la tabla completa con estratificación
   multietiqueta iterativa (Security + subcategorías, semilla 42). La tarea 1 usa todos los
   requisitos de cada fold y la tarea 2 solo los de seguridad, así que en el fold k el modelo de
   subcategorías no vio ningún requisito que el modelo binario evalúa en ese fold. **[Decisión propia]**
2. **Folds internos:** 3 por fold externo y por tarea, estratificados sobre las etiquetas de la tarea.
3. **Optuna** (TPE + poda por mediana): maximiza la media del objetivo en los folds internos con
   umbral 0,50. El trial 0 es la configuración por defecto. Trials por fold externo: 25 para CNN y
   BiLSTM, 12 para los Transformers.
4. **Parada temprana:** un fold se detiene cuando pasan `PATIENCE` épocas sin que mejore el objetivo
   de validación ni la pérdida de validación.

   | | CNN | BiLSTM | Transformers |
   |---|---|---|---|
   | Máximo de épocas, tarea 1 | 50 | 50 | 15 |
   | Máximo de épocas, tarea 2 | 50 | 50 | 30 |
   | Paciencia, tarea 1 | 10 | 8 | 3 |
   | Paciencia, tarea 2 | 10 | 8 | 5 |

5. **Umbrales por etiqueta** (0,40–0,60): con las predicciones fuera de fold del mejor trial.
6. **Modelo final del fold externo:** se entrena con todo el entrenamiento externo durante la media
   de las mejores épocas internas y predice una vez el fold de prueba.
7. **Métricas:**
   - tarea 1: F1, precisión y recall de la clase seguridad, y macro-F1 de las dos clases (objetivo: F1;
     desempate: recall);
   - tarea 2: macro-F1, weighted-F1, micro-F1 y macro-recall (objetivo: macro-F1; desempate: macro-recall);
   - cascada: las métricas de la tarea 2 sobre todos los requisitos, comparadas con la tarea 2 sola.

### Hiperparámetros

| Modelo | Por defecto | Búsqueda |
|---|---|---|
| CNN | emb 64, filtros 100, kernel 3, batch 32, dropout 0,2, lr 1e-3 | emb {64,100,128}, filtros {100,128}, kernel {3,4,5}, batch {16,32}, dropout {0,2–0,5}, lr 1e-4–1e-2 |
| BiLSTM | emb 100, hidden 64, 1 capa, batch 32, dropout 0,2, lr 1e-3 | emb {64,100,128}, hidden {32,64,128}, capas {1,2}, batch {16,32,50}, dropout {0,2; 0,3; 0,5}, lr 5e-4–2e-3 |
| BETO / RoBERTa-BNE / DistilBETO | lr 2e-5, batch 16, dropout 0,1, weight decay 0,01, AdamW | lr 1e-5–5e-5, batch {8,16}, dropout 0,1–0,5, weight decay 1e-8–0,3, {AdamW, Adam}, warmup {0; 0,06; 0,1} |

Todos buscan también `pos_weight` (BCE con el término positivo ponderado por `sqrt(N_neg/N_pos)` del
entrenamiento). Las épocas ya no se buscan: las fija la parada temprana. Los Transformers usan
`max_length` 128 y solo la cabeza toma el dropout ajustado. RoBERTa-BNE se carga desde el espejo
`Buzzeitor/roberta-base-bne`, porque el repositorio oficial ya no publica los pesos.

## Archivos de cada corrida (`outputs/output_raw/<tarea>/<condición>/<dataset>/<modelo>/`)

| Archivo | Contenido |
|---|---|
| `summary.json` | resumen: folds externos, métricas agrupadas y configuración ganadora de cada fold |
| `outer_fold_metrics.csv`, `pooled_metrics_by_label.csv` | métricas por fold externo y por etiqueta |
| `pooled_outer_test_predictions.csv` | predicciones de los 5 folds de prueba juntas |
| `pooled_confusion_by_label.csv` | VP, FP, FN y VN de cada etiqueta |
| `outer_<k>/optuna_trials.csv`, `optuna_study.db`, `trials/` | estudio de Optuna del fold externo (reanudable) |
| `outer_<k>/inner_cv_history.csv`, `inner_cv_curves.png` | pérdida y objetivo por época de los folds internos del mejor trial |
| `outer_<k>/outer_test_predictions.csv`, `outer_result.json` | predicción del fold de prueba y su resumen |
| `outer_<k>/nonsecurity_predictions.csv` | solo tarea 2: requisitos sin seguridad del fold de prueba, para la cascada |

`results_<tarea>_<condición>.xlsx` tiene las hojas: resumen general, F1 por clase, métricas por
clase, matrices de confusión, folds externos, Scott-Knott ESD, Cochran Q y McNemar, confusión entre
subcategorías (tarea 2), errores para revisión, mejor modelo, Transformers vs. DL y, en C1, diversidad
de los sintéticos.

### Pruebas entre modelos

- **Scott-Knott ESD** no paramétrico (Binkhonain y Alfayez, 2025) sobre el objetivo de los 5 folds
  externos; los grupos con |δ de Cliff| < 0,147 (umbral del paquete de R ScottKnottESD) se fusionan.
- **Q de Cochran + McNemar exacto con Holm** por dataset y etiqueta, sobre las predicciones agrupadas.
- **Mejor modelo:** dentro del grupo superior de Scott-Knott ESD, el de mayor desempate agrupado
  (recall de seguridad en la tarea 1, macro-recall en la tarea 2).
- **C0 vs. C1** (`scripts/compare_conditions.py`, tarea 2): McNemar por etiqueta, bootstrap pareado
  del delta de macro-F1 y Wilcoxon sobre los pares (dataset, modelo).
