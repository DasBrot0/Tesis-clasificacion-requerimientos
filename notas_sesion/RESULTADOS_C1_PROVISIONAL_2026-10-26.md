# Resultados de C1 (reales + sintéticos), corrida provisional (2026-10-26 y 2026-10-27)

Logs: `logs/run_all_c1_augmented.log`, `logs/run_all_c1_augmented_bilstm.log`,
`logs/run_all_c1_extension.log` y `logs/c1_augmented_<dataset>_<modelo>.log`.
Tabla: `artifacts/c1_augmented/model_comparison.csv`.

**Qué se entrenó:** condición `c1_augmented`, los cuatro modelos en los tres datasets.
**Protocolo:** CV anidada 5 externos × 3 internos, umbrales por etiqueta con el OOF interno, cinco
salidas (Security + 4 subcategorías). Trials por fold externo, en dos etapas (`PLAN_AUMENTO_DATOS.md`,
sección 0):

| Modelo | Etapa 1 | Etapa 2 (objetivo) | Estado al 27-10 |
|---|---:|---:|---|
| BETO, RoBERTa-BNE | 6 | 12 | **completos con 12** |
| CNN, BiLSTM | 15 | 25 | **con 15**; la extensión a 25 queda pendiente (se cortó la cola por tiempo) |

Cada etapa terminada queda archivada en `artifacts/c1_augmented/<dataset>/<modelo>/stage_<n>trials/`.

**Datos:** etiquetas provisionales (voto de Sonnet en PROMISE y SecReq; conversión sin auditar en
DOSSPRE), traducción DeepL + glosario provisional y 321 sintéticos de la versión 2 del filtro
(`PLAN_AUMENTO_DATOS.md`, sección 6), sin revisión experta.
**Sus resultados son provisionales y no son los finales de la tesis.**

## 1. Ejecución

| Dataset | CNN (15) | BiLSTM (15) | BETO (6 + 6) | RoBERTa-BNE (6 + 6) |
|---|---:|---:|---:|---:|
| PROMISE | 8,4 min | 24,0 min | 47,9 + 56,6 min | 49,1 + 50,3 min |
| SecReq | 8,6 min | 26,8 min | 52,7 + 63,4 min | 53,7 + 65,1 min |
| DOSSPRE | 18,8 min | 53,1 min | 103,6 + 118,0 min | 96,5 + 92,6 min |

Los tres BiLSTM figuran como `ERROR` en `run_all`, pero no hay traceback: cada log termina con
todas las métricas y `summary.json` se escribió en el mismo segundo en que terminó el proceso. Es el
cierre nativo de cuDNN/LSTM en Windows ya visto en la corrida del 19-10 (0xC0000409), posterior al
guardado. **Los resultados de BiLSTM son válidos.**

## 2. Resultados

Métricas sobre los folds externos (todos reales). "Agrupado" = las predicciones de los 5 externos
juntas, cada requisito una vez con los umbrales de su externo.

| Dataset | Modelo | Trials | Macro-F1 (5 externos) | Macro-F1 agrupado | Weighted-F1 agrupado | Micro-F1 agrupado | Precisión / recall Security |
|---|---|---:|---|---:|---:|---:|---|
| PROMISE | CNN | 15 | 0,316 ± 0,092 | 0,346 | 0,436 | 0,463 | 0,50 / 0,68 |
| PROMISE | BiLSTM | 15 | 0,329 ± 0,112 | 0,352 | 0,456 | 0,464 | 0,57 / 0,63 |
| PROMISE | BETO | 12 | **0,618 ± 0,070** | **0,644** | 0,707 | 0,710 | 0,79 / 0,80 |
| PROMISE | RoBERTa-BNE | 12 | 0,604 ± 0,116 | 0,619 | 0,697 | 0,693 | 0,78 / 0,86 |
| SecReq | CNN | 15 | 0,592 ± 0,035 | 0,605 | 0,699 | 0,706 | 0,70 / 0,89 |
| SecReq | BiLSTM | 15 | 0,428 ± 0,062 | 0,444 | 0,622 | 0,640 | 0,75 / 0,82 |
| SecReq | BETO | 12 | 0,620 ± 0,068 | 0,635 | 0,775 | 0,781 | 0,86 / 0,89 |
| SecReq | RoBERTa-BNE | 12 | **0,666 ± 0,098** | **0,671** | 0,774 | 0,780 | 0,87 / 0,88 |
| DOSSPRE | CNN | 15 | 0,519 ± 0,055 | 0,521 | 0,575 | 0,579 | 0,62 / 0,67 |
| DOSSPRE | BiLSTM | 15 | 0,470 ± 0,054 | 0,472 | 0,551 | 0,557 | 0,64 / 0,64 |
| DOSSPRE | BETO | 12 | **0,667 ± 0,026** | **0,670** | 0,716 | 0,717 | 0,73 / 0,79 |
| DOSSPRE | RoBERTa-BNE | 12 | 0,663 ± 0,013 | 0,669 | 0,712 | 0,713 | 0,78 / 0,74 |

### F1 agrupado por etiqueta

| Dataset | Modelo | Security | Confidentiality | Integrity | Authenticity | Resistance |
|---|---|---:|---:|---:|---:|---:|
| PROMISE (79 / 37 / 14 / 18 / 10) | CNN | 0,58 | 0,32 | 0,12 | 0,33 | 0,38 |
| | BiLSTM | 0,60 | 0,34 | 0,13 | 0,40 | 0,29 |
| | BETO | 0,79 | 0,64 | 0,46 | 0,73 | 0,60 |
| | RoBERTa-BNE | 0,82 | 0,60 | 0,36 | 0,65 | 0,67 |
| SecReq (177 / 31 / 60 / 70 / 12) | CNN | 0,78 | 0,50 | 0,61 | 0,70 | 0,43 |
| | BiLSTM | 0,78 | 0,28 | 0,42 | 0,63 | 0,11 |
| | BETO | 0,88 | 0,53 | 0,68 | 0,79 | 0,30 |
| | RoBERTa-BNE | 0,87 | 0,53 | 0,61 | 0,82 | 0,52 |
| DOSSPRE (406 / 123 / 33 / 92 / 120) | CNN | 0,65 | 0,52 | 0,44 | 0,54 | 0,46 |
| | BiLSTM | 0,64 | 0,50 | 0,32 | 0,48 | 0,42 |
| | BETO | 0,76 | 0,69 | 0,55 | 0,70 | 0,65 |
| | RoBERTa-BNE | 0,76 | 0,66 | 0,55 | 0,75 | 0,63 |

Entre paréntesis, los positivos reales por etiqueta (S / C / I / A / R).

### Efecto de pasar de 6 a 12 trials en los transformers (macro-F1 agrupado)

| Dataset | BETO 6 → 12 | RoBERTa-BNE 6 → 12 |
|---|---|---|
| PROMISE | 0,567 → 0,644 (+0,08) | 0,583 → 0,619 (+0,04) |
| SecReq | 0,623 → 0,635 (+0,01) | 0,685 → 0,671 (−0,01) |
| DOSSPRE | 0,649 → 0,670 (+0,02) | 0,646 → 0,669 (+0,02) |

## 3. Observaciones

- **Los transformers son claramente mejores** (macro-F1 agrupado 0,62 a 0,67) que CNN y BiLSTM
  (0,35 a 0,61). BETO y RoBERTa-BNE quedan a menos de 0,03 entre sí: BETO es el mejor en PROMISE y
  DOSSPRE; RoBERTa-BNE, en SecReq.
- **Duplicar los trials ayuda sobre todo en PROMISE** (+0,04 a +0,08), el dataset más pequeño; en
  SecReq y DOSSPRE el cambio es de ±0,02.
- **BiLSTM es el peor o empata con CNN.** En SecReq queda 16 puntos por debajo de CNN y casi no
  detecta Resistance (F1 0,11 con 12 positivos). CNN y BiLSTM todavía tienen 15 trials, no 25.
- **Sesgo hacia Security** (todos los sintéticos tienen `Security=1`): los transformers mantienen
  la precisión cerca del recall (0,73 a 0,87). CNN la tiene por debajo en PROMISE (0,50 frente a
  0,68) y en SecReq (0,70 frente a 0,89), lo que sugiere sobrepredicción. Solo se confirma
  comparando con C0.
- **`pos_weight=True` gana en 54 de los 60 folds externos.** La ponderación de la pérdida se suma
  al aumento.
- **Las etiquetas más débiles son las de menos positivos reales:** Integrity en PROMISE (14; F1 de
  0,12 a 0,46) y Resistance en SecReq (12; de 0,11 a 0,52), aunque son de las más aumentadas.
- **Variabilidad entre externos:** desviación de 0,07 a 0,12 en PROMISE y SecReq, y de 0,01 a 0,03
  en los transformers de DOSSPRE, el dataset más grande.

## 4. Comparabilidad

- **No se puede concluir nada sobre el aumento** hasta entrenar C0 con el mismo protocolo y las
  mismas dos etapas de trials, y ejecutar `python scripts/compare_conditions.py`.
- Antes de C0 conviene completar C1: extender CNN y BiLSTM a 25 trials
  (`python scripts/extend_trials.py --condition c1_augmented --models cnn bilstm` y luego
  `python run_all_training.py --condition c1_augmented --models cnn bilstm --n-trials 25`, ~1,5 h).
- La corrida del 19-10 (`old/run_2026-10-19_sonnet_deepl_holdout/`) **no sirve como C0**: usaba un
  test fijo del 15 %, siete etiquetas, DOSSPRE con Availability y Maintainability en `Security=1`
  y 30/12 trials.
