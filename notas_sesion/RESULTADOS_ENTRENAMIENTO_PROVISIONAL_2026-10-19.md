# Resultados del entrenamiento provisional (2026-10-18 22:32 a 2026-10-19 06:19)

Log: `logs/provisional_training_20261018_223200.log`. Duración total aprox. 7 h 47 min. Etiquetas provisionales de Sonnet + traducción DeepL con glosario + dropout de RoBERTa corregido. Optuna + CV de 5 folds.

## 1. Ejecución por par

| Dataset | Modelo | Código | Duración |
|---|---|---|---|
| promise | cnn | 0 | 6.5 min |
| promise | bilstm | 3221226505 (0xC0000409) | 19.9 min |
| promise | beto | 0 | 45.2 min |
| promise | roberta_bne | 0 | 51.9 min |
| secreq | cnn | 0 | 5.6 min |
| secreq | bilstm | 3221226505 | 17.9 min |
| secreq | beto | 0 | 49.2 min |
| secreq | roberta_bne | 0 | 51.9 min |
| dosspre | cnn | 0 | 11.9 min |
| dosspre | bilstm | 3221226505 | 32.8 min |
| dosspre | beto | 0 | 83.7 min |
| dosspre | roberta_bne | 0 | 90.6 min |

Nota: los 3 BiLSTM terminan con 0xC0000409 (crash nativo de Windows, "stack buffer overrun", típico de cuDNN/LSTM al cerrar el proceso). No hay traceback ni OOM en el log. El log y los artefactos (summary.json, test_metrics_by_label.csv, model_comparison.csv) se escribieron completos antes del crash (el log imprime las métricas de test y "Modelo final"), así que los resultados BiLSTM parecen válidos. Conviene confirmarlo (el crash ocurre tras guardar, según las marcas de hora de los archivos).

## 2. Resultados nuevos (Sonnet + DeepL)

Etiquetas excluidas del promedio macro: PROMISE y SecReq = Non_repudiation, Accountability (sin soporte); DOSSPRE = Non_repudiation.

| Dataset | Modelo | CV macro-F1 (media ± std) | Test macro-F1 | Test weighted-F1 | Test recall Security |
|---|---|---|---|---|---|
| PROMISE | CNN | 0.4836 ± 0.1014 | 0.315 | 0.452 | 0.385 |
| PROMISE | BiLSTM | 0.4485 ± 0.0558 | 0.124 | 0.242 | 0.231 |
| PROMISE | BETO | 0.6966 ± 0.0928 | 0.556 | 0.720 | 0.846 |
| PROMISE | RoBERTa-BNE | 0.7272 ± 0.1435 | 0.454 | 0.668 | 0.692 |
| SecReq | CNN | 0.6187 ± 0.1005 | 0.674 | 0.723 | 0.778 |
| SecReq | BiLSTM | 0.5432 ± 0.0740 | 0.502 | 0.618 | 0.778 |
| SecReq | BETO | 0.6751 ± 0.0356 | 0.556 | 0.767 | 0.963 |
| SecReq | RoBERTa-BNE | 0.6774 ± 0.0470 | 0.758 | 0.839 | 0.963 |
| DOSSPRE | CNN | 0.5495 ± 0.0405 | 0.516 | 0.609 | 0.693 |
| DOSSPRE | BiLSTM | 0.4711 ± 0.0359 | 0.362 | 0.500 | 0.653 |
| DOSSPRE | BETO | 0.6110 ± 0.0689 | 0.624 | 0.739 | 0.893 |
| DOSSPRE | RoBERTa-BNE | 0.6357 ± 0.0677 | 0.636 | 0.713 | 0.920 |

## 3. Corrida anterior (regex + opus-mt), `old/run_2026-10_regex_opusmt/artifacts/`

| Dataset | Modelo | CV macro-F1 | Test macro-F1 | Test weighted-F1 | Test recall Security |
|---|---|---|---|---|---|
| PROMISE | CNN | 0.6069 ± 0.0989 | 0.444 | 0.571 | 0.583 |
| PROMISE | BiLSTM | 0.5565 ± 0.1205 | 0.511 | 0.542 | 0.917 |
| PROMISE | BETO | 0.8296 ± 0.0614 | 0.738 | 0.722 | 0.833 |
| SecReq | CNN | 0.6118 ± 0.0379 | 0.556 | 0.743 | 0.926 |
| SecReq | BiLSTM | 0.5144 ± 0.0817 | 0.427 | 0.599 | 0.741 |
| SecReq | BETO | 0.6886 ± 0.0302 | 0.532 | 0.705 | 0.815 |
| DOSSPRE | CNN | 0.5148 ± 0.0419 | 0.423 | 0.554 | 0.671 |
| DOSSPRE | BiLSTM | 0.4392 ± 0.0307 | 0.346 | 0.448 | 0.447 |
| DOSSPRE | BETO | 0.6383 ± 0.0465 | 0.607 | 0.678 | 0.671 |

RoBERTa-BNE no aparece en la corrida anterior (no se entrenó o falló). Además, en PROMISE la corrida anterior promediaba solo 3 etiquetas (Security, Confidentiality, Authenticity); la nueva promedia 5.

**Advertencia de comparabilidad:** las métricas NO son estrictamente comparables. Cambiaron las etiquetas (regex -> Sonnet provisional), la traducción (opus-mt -> DeepL con glosario) y el conjunto de etiquetas incluidas en el promedio macro (en PROMISE pasó de 3 a 5), además del dropout de RoBERTa. Las diferencias reflejan sobre todo el cambio de etiquetas, no una mejora o empeoramiento del modelo.

## 4. Anomalías y observaciones

- RoBERTa-BNE funciona bien tras el arreglo del dropout (macro-F1 de CV 0.64-0.73), sin colapso a ~0. Es el mejor o empata con BETO en CV en los 3 datasets, y el mejor en test en SecReq (0.758) y DOSSPRE (0.636).
- BiLSTM es el peor modelo en los 3 datasets. En PROMISE el test macro-F1 es 0.124 (weighted 0.242, recall de Security 0.231), muy por debajo de su CV (0.449): casi colapsó en test. Además, crash 0xC0000409 en los 3 BiLSTM.
- Alta desviación en PROMISE: RoBERTa-BNE 0.1435, CNN 0.1014, BETO 0.0928. Es el dataset más pequeño y con pocas etiquetas minoritarias.
- Brecha CV-test grande en PROMISE (RoBERTa 0.727 vs 0.454; BETO 0.697 vs 0.556; CNN 0.484 vs 0.315). El test de PROMISE es muy pequeño (p. ej. Integrity 2, Resistance 1 ejemplos; Resistance con F1 = 0 cuenta en el promedio), por lo que el macro-F1 de test es muy inestable.
- Accountability (PROMISE/SecReq) y Non_repudiation (todos) están excluidas del promedio macro por falta de soporte en test.
- CV con config. por defecto (umbral 0.5) es mucho menor que con Optuna (p. ej. BETO PROMISE 0.263 vs 0.682); esas columnas están en `artifacts/model_comparison.csv`.
- Tiempos: los transformers dominan (BETO/RoBERTa 45-91 min por par); DOSSPRE es el más lento.

Archivos: `artifacts/model_comparison.csv`, `artifacts/<dataset>/<modelo>/summary.json`, `test_metrics_by_label.csv`.
