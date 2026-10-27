# Resultados de C1 (reales + sintéticos), corrida provisional (2026-10-26 00:08 a 05:48)

Logs: `logs/run_all_c1_augmented.log`, `logs/run_all_c1_augmented_bilstm.log` y
`logs/c1_augmented_<dataset>_<modelo>.log`. Tabla: `artifacts/c1_augmented/model_comparison.csv`.

**Qué se entrenó:** condición `c1_augmented`, modelos CNN, BiLSTM y RoBERTa-BNE (BETO quedó fuera
por tiempo de cómputo) en los tres datasets.
**Protocolo:** CV anidada 5 externos × 3 internos, Optuna con 15 trials por externo (CNN, BiLSTM) y
6 (RoBERTa-BNE), umbrales por etiqueta con el OOF interno, cinco salidas (Security + 4
subcategorías).
**Datos:** etiquetas provisionales (voto de Sonnet en PROMISE y SecReq; conversión sin auditar en
DOSSPRE), traducción DeepL + glosario provisional y 321 sintéticos de la versión 2 del filtro
(`PLAN_AUMENTO_DATOS.md`, sección 6), sin revisión experta.
**Sus resultados son provisionales y no son los finales de la tesis.**

## 1. Ejecución

| Dataset | CNN | BiLSTM | RoBERTa-BNE |
|---|---:|---:|---:|
| PROMISE | 8,4 min | 24,0 min | 49,1 min |
| SecReq | 8,6 min | 26,8 min | 53,7 min |
| DOSSPRE | 18,8 min | 53,1 min | 96,5 min |

Los tres BiLSTM figuran como `ERROR` en `run_all`, pero no hay traceback: cada log termina con
todas las métricas y `summary.json` se escribió en el mismo segundo en que terminó el proceso. Es el
cierre nativo de cuDNN/LSTM en Windows ya visto en la corrida del 19-10 (0xC0000409), posterior al
guardado. **Los resultados de BiLSTM son válidos.**

## 2. Resultados

Métricas sobre los folds externos (todos reales). "Agrupado" = las predicciones de los 5 externos
juntas, cada requisito una vez con los umbrales de su externo.

| Dataset | Modelo | Macro-F1 (5 externos) | Macro-F1 agrupado | Weighted-F1 agrupado | Micro-F1 agrupado | Precisión / recall Security |
|---|---|---|---:|---:|---:|---|
| PROMISE | CNN | 0,316 ± 0,092 | 0,346 | 0,436 | 0,463 | 0,50 / 0,68 |
| PROMISE | BiLSTM | 0,329 ± 0,112 | 0,352 | 0,456 | 0,464 | 0,57 / 0,63 |
| PROMISE | RoBERTa-BNE | **0,565 ± 0,090** | **0,583** | 0,675 | 0,671 | 0,78 / 0,81 |
| SecReq | CNN | 0,592 ± 0,035 | 0,605 | 0,699 | 0,706 | 0,70 / 0,89 |
| SecReq | BiLSTM | 0,428 ± 0,062 | 0,444 | 0,622 | 0,640 | 0,75 / 0,82 |
| SecReq | RoBERTa-BNE | **0,675 ± 0,097** | **0,685** | 0,786 | 0,788 | 0,86 / 0,86 |
| DOSSPRE | CNN | 0,519 ± 0,055 | 0,521 | 0,575 | 0,579 | 0,62 / 0,67 |
| DOSSPRE | BiLSTM | 0,470 ± 0,054 | 0,472 | 0,551 | 0,557 | 0,64 / 0,64 |
| DOSSPRE | RoBERTa-BNE | **0,643 ± 0,036** | **0,646** | 0,698 | 0,700 | 0,75 / 0,76 |

### F1 agrupado por etiqueta

| Dataset | Modelo | Security | Confidentiality | Integrity | Authenticity | Resistance |
|---|---|---:|---:|---:|---:|---:|
| PROMISE (79 / 37 / 14 / 18 / 10) | CNN | 0,58 | 0,32 | 0,12 | 0,33 | 0,38 |
| | BiLSTM | 0,60 | 0,34 | 0,13 | 0,40 | 0,29 |
| | RoBERTa-BNE | 0,80 | 0,60 | 0,38 | 0,62 | 0,53 |
| SecReq (177 / 31 / 60 / 70 / 12) | CNN | 0,78 | 0,50 | 0,61 | 0,70 | 0,43 |
| | BiLSTM | 0,78 | 0,28 | 0,42 | 0,63 | 0,11 |
| | RoBERTa-BNE | 0,86 | 0,61 | 0,65 | 0,85 | 0,46 |
| DOSSPRE (406 / 123 / 33 / 92 / 120) | CNN | 0,65 | 0,52 | 0,44 | 0,54 | 0,46 |
| | BiLSTM | 0,64 | 0,50 | 0,32 | 0,48 | 0,42 |
| | RoBERTa-BNE | 0,75 | 0,66 | 0,51 | 0,71 | 0,61 |

Entre paréntesis, los positivos reales por etiqueta (S / C / I / A / R).

## 3. Observaciones

- **RoBERTa-BNE es el mejor modelo en los tres datasets** (macro-F1 agrupado 0,58 a 0,69), con
  ventaja sobre CNN de 8 a 24 puntos; la mayor, en PROMISE, el dataset más pequeño.
- **BiLSTM es el peor o empata con CNN.** En SecReq queda 16 puntos por debajo de CNN y casi no
  detecta Resistance (F1 0,11 con 12 positivos).
- **Sesgo hacia Security** (todos los sintéticos tienen `Security=1`): RoBERTa-BNE mantiene la
  precisión igual o mayor que el recall en los tres datasets. CNN la tiene por debajo en PROMISE
  (0,50 frente a 0,68) y en SecReq (0,70 frente a 0,89), lo que sugiere sobrepredicción. Solo se
  confirma comparando con C0.
- **`pos_weight=True` gana en 38 de los 45 folds externos**, y en todos los de CNN en SecReq y DOSSPRE.
  La ponderación de la pérdida se suma al aumento.
- **Las etiquetas más débiles son las de menos positivos reales:** Integrity en PROMISE (14; F1 de
  0,12 a 0,38) y Resistance en SecReq (12; de 0,11 a 0,46), aunque son de las más aumentadas.
- **Variabilidad entre externos:** desviación de 0,09 a 0,11 en PROMISE y en RoBERTa-BNE de SecReq.
  Con 92 requisitos por fold externo en PROMISE, un positivo raro más o menos mueve mucho el F1.

## 4. Comparabilidad

- **No se puede concluir nada sobre el aumento** hasta entrenar C0 con el mismo protocolo
  (`python run_all_training.py --condition c0_real --models cnn bilstm roberta_bne`, ~5,5 h) y
  ejecutar `python scripts/compare_conditions.py`.
- La corrida del 19-10 (`old/run_2026-10-19_sonnet_deepl_holdout/`) **no sirve como C0**: usaba un
  test fijo del 15 %, siete etiquetas, DOSSPRE con Availability y Maintainability en `Security=1`
  y 30/12 trials.
