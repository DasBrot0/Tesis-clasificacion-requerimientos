# Clasificación multietiqueta de requisitos de seguridad

Cuatro modelos (CNN, BiLSTM, BETO, RoBERTa-BNE) entrenados **por separado** sobre tres datasets
(PROMISE_exp, SecReq, DOSSPRE), en dos condiciones: **C0** (`c0_real`, solo requisitos reales) y
**C1** (`c1_augmented`, reales + sintéticos generados con LLM, solo en entrenamiento). Cada modelo
predice `Security` y cuatro subcategorías de seguridad de ISO/IEC 25010 (`Confidentiality`,
`Integrity`, `Authenticity`, `Resistance`). `Non_repudiation` y `Accountability` se descartaron el
25-10-2026 por falta de datos.

## Estructura

```
datasets/
  original/     archivos fuente, sin modificar
  interim/      glosario, traducciones, preanotaciones y aumento (augmentation/)
  training/     promise.csv, secreq.csv, dosspre.csv  <- los datos reales que leen los modelos
src/
  config.py     rutas, etiquetas, condiciones, hiperparámetros y espacios de búsqueda de Optuna
  data.py       carga validada, CV anidada y unión de sintéticos por fold
  metrics.py    métricas multietiqueta y umbrales por etiqueta
  models.py     arquitecturas y codificación del texto
  training.py   rutina de entrenamiento
  tuning.py     estudio de Optuna sobre los folds internos
  pipeline.py   corrida completa de un trío (condición, dataset, modelo)
scripts/
  prepare_datasets.py      original/ -> training/
  verify_subcategories.py  cobertura de las subcategorías
  augmentation/            flujo de PLAN_AUMENTO_DATOS.md (G1, G3, G4, G5-G7)
train.py                   una corrida: --condition C --dataset X --model Y
run_all_training.py        las corridas de una condición
tests/                     pruebas rápidas del protocolo (no entrenan)
artifacts/splits/                         folds externos e internos y distribución de etiquetas
artifacts/<condición>/<dataset>/<modelo>/ métricas, trials, predicciones, curvas
artifacts/<condición>/model_comparison.csv
logs/                      salida de cada corrida
old/                       código y corridas anteriores (ver old/run_2026-10-19_sonnet_deepl_holdout/)
```

## Uso

```powershell
pip install -r requirements.txt
python scripts/prepare_datasets.py       # reconstruye datasets/training/
python scripts/verify_subcategories.py
python -m unittest discover -s tests -v
python run_all_training.py --condition c1_augmented --models cnn roberta_bne
python train.py --condition c0_real --dataset secreq --model beto
```

Una corrida interrumpida se retoma desde el último fold externo terminado. `--save-checkpoints`
guarda el modelo de cada fold externo en `checkpoints/` (no se guarda por defecto).

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

`datasets/interim/augmentation/synthetic_<dataset>.csv`: 92 (PROMISE), 90 (SecReq) y 139 (DOSSPRE)
requisitos generados por `claude-sonnet-5-5`, traducidos con DeepL + glosario y verificados a
ciegas por 3 corridas del mismo modelo (corrida provisional, sin revisión experta). Cada uno guarda
su `Anchor_ID`: el requisito real del que se generó. Detalle en
[`notas_sesion/PLAN_AUMENTO_DATOS.md`](notas_sesion/PLAN_AUMENTO_DATOS.md).

## Protocolo (por dataset, idéntico para los cuatro modelos y las dos condiciones)

1. **Folds externos**: 5 folds con estratificación multietiqueta iterativa (semilla 42). Cada
   requisito real se evalúa exactamente una vez.
2. **Folds internos**: dentro del entrenamiento de cada fold externo, 3 folds para Optuna.
3. **Optuna** (TPE + poda por mediana): el objetivo es el macro-F1 medio de los folds internos con
   umbral 0.50. El trial 0 siempre es la configuración por defecto. Trials por fold externo: 15
   para CNN y BiLSTM, 6 para BETO y RoBERTa-BNE.
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

## Artefactos de cada corrida (`artifacts/<condición>/<dataset>/<modelo>/`)

| Archivo | Contenido |
|---|---|
| `summary.json` | resumen completo: folds externos, métricas agrupadas, configuración por fold |
| `outer_fold_metrics.csv` | métricas de cada fold externo |
| `pooled_outer_test_predictions.csv`, `pooled_metrics_by_label.csv` | predicciones externas agrupadas y F1 por etiqueta |
| `outer_<k>/outer_result.json` | configuración ganadora, umbrales y prueba del fold externo k |
| `outer_<k>/optuna_trials.csv`, `optuna_study.db`, `trials/` | estudio de Optuna del fold externo (reanudable) |
| `outer_<k>/inner_cv_history.csv`, `inner_cv_curves.png` | curvas internas del ganador |
| `outer_<k>/outer_test_predictions.csv` | predicciones del fold externo de prueba |
