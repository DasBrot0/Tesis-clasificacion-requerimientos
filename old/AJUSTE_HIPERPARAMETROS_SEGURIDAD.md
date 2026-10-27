# Registro de análisis y ajuste — modelos de Seguridad

Fecha: 2026-09-28  
Semilla: 42  
Partición común: 629 train / 135 validation / 135 test  
Regla: test permanece sin consultar; todas las decisiones usan exclusivamente validation.

## Línea base

| Modelo | Configuración original | Mejor época | Val macro-F1 | Val recall Security |
|---|---|---:|---:|---:|
| CNN | emb=64, filtros=100, kernel=3, batch=32, lr=0.001, dropout=0.2, 50 épocas, umbral=0.50 | 47 | 0.2766 | 0.6410 |
| BiLSTM | emb=100, hidden=64, 1 capa, batch=32, lr=0.001, dropout=0.2, 30 épocas, umbral=0.50 | 30 | 0.1825 | 0.5128 |
| BETO | lr=2e-5, batch=16, 5 épocas, dropout=0.1, wd=0.01, max_len=128, umbral=0.50 | 5 | 0.3219 | 0.6923 |
| RoBERTa-BNE | igual a BETO; mirror explícito de los pesos retirados | 1 | 0.0000 | 0.0000 |

## Diagnóstico

1. **Desbalance extremo.** En train hay 179 positivos de Security, pero solo 6 de Non-repudiation, 8 de Accountability y 15 de Resistance. En validation esas etiquetas tienen 1, 2 y 4 positivos. Macro-F1 queda dominado por estimaciones muy inestables de clases raras.
2. **BCE sin ponderación favorece el negativo.** Non-repudiation y Accountability representan menos del 1.3 % de train. Predecir siempre cero reduce la pérdida, aunque produce F1=0.
3. **El umbral no es la causa principal.** Barrer 0.40–0.60 sobre las predicciones existentes mejora BETO solo de 0.3219 a 0.3303 (umbral 0.40); CNN y BiLSTM cambian muy poco. RoBERTa permanece en 0 porque todas sus probabilidades están por debajo de 0.40.
4. **Subajuste de RoBERTa.** Su train loss baja de 0.4411 a 0.2388, pero las probabilidades quedan casi constantes y cercanas a las prevalencias. Cinco épocas con lr=2e-5 no separan positivos de negativos en este checkpoint.
5. **Etiquetas provisionales.** Las seis subcategorías positivas son preanotaciones heurísticas (`auto_prelabel`), no consenso experto. Esto limita la interpretación sustantiva aun cuando mejore la métrica.

## Experimento 1 — búsqueda controlada dentro de la metodología

Se activa `RUN_HPARAM_SEARCH=True` solo durante la ejecución. No se modifica `INITIAL_CONFIG`, no se usa test y no se introduce todavía ponderación de clases. Se evalúan las listas `SEARCH_TRIALS` ya declaradas en cada notebook.

Cambios respecto del original:

- CNN trial 1: emb 100, filtros 128, kernel 4, batch 16, dropout 0.3, lr 0.0005, umbral 0.45.
- CNN trial 2: emb 128, kernel 5, 60 épocas, dropout 0.5, lr 0.005, umbral 0.55.
- BiLSTM trial 1: emb 64, hidden 32, 2 capas, batch 16, 20 épocas, dropout 0.3, lr 0.0005, umbral 0.45.
- BiLSTM trial 2: emb 128, hidden 128, batch 50, 50 épocas, dropout 0.5, lr 0.002, umbral 0.55.
- BETO/RoBERTa trial 1: lr 1e-5, batch 8, 10 épocas, dropout 0.2, wd 1e-4, max_len 256, umbral 0.45.
- BETO/RoBERTa trial 2: lr 5e-5, batch 16, 15 épocas, dropout 0.5, wd 0.1, max_len 128, umbral 0.55.

Resultados parciales:

- CNN: ningún trial superó la línea base (macro-F1 0.2766).
- BiLSTM: el trial 2 mejoró macro-F1 de 0.1825 a 0.2575 y recall de Security de 0.5128 a 0.5641.
- BETO: búsqueda interrumpida por decisión del usuario durante la repetición del trial base; no hay conclusión nueva.
- RoBERTa-BNE: búsqueda aún no iniciada.

Estado: **pausado antes de continuar moviendo hiperparámetros**. La evidencia indica que el siguiente experimento debe atacar primero el desbalance, manteniendo test reservado.

## Experimento 2 — nueva partición y aumento train-only

Solicitado por el usuario después de revisar la falta de positivos también en validation/test.

Cambios respecto del protocolo anterior:

1. Se descartaron todos los checkpoints y resultados de los entrenamientos anteriores.
2. Se reemplazó el split iterativo por una asignación entera determinista con semilla 42 y tamaños idénticos 629/135/135. Mantiene en validation y test, respectivamente: Non-repudiation=2, Accountability=3 y Resistance=5.
3. Train real conserva Non-repudiation=5, Accountability=6 y Resistance=12. Ningún texto de validation/test se usa como semilla.
4. Se generan variantes únicamente desde filas reales de train con una sola subcategoría positiva. Solo se modifica el marco lingüístico del requisito; se conserva el texto semántico central y todas sus etiquetas.
5. Se añadieron 201 filas sintéticas únicas. Las seis subcategorías quedan exactamente con 62 positivos cada una en train aumentado (830 filas). Security pasa de 178 a 379 positivos porque engloba todas las subcategorías.
6. Validation y test permanecen 100 % reales y sin aumento.
7. El umbral global se selecciona entre 0.40 y 0.60 usando solo validation, con macro-F1 y recall de Security como desempate. Test continúa bloqueado.
8. Se vuelve a ejecutar la búsqueda controlada declarada en cada notebook. `INITIAL_CONFIG` no se altera.

### Resultados finales del experimento 2 (validation)

| Modelo | Trial ganador | Época | Umbral ajustado | Macro-F1 | Recall Security | F1 Security |
|---|---|---:|---:|---:|---:|---:|
| BETO | configuración inicial | 5 | 0.40 | **0.4976** | **0.9487** | **0.8605** |
| BiLSTM | emb=128, hidden=128, batch=50, lr=0.002, dropout=0.5 | 41 | 0.45 | 0.3614 | 0.7949 | 0.8158 |
| CNN | emb=128, filtros=100, kernel=5, lr=0.005, dropout=0.5 | 51 | 0.575 | 0.3442 | 0.8205 | 0.8000 |
| RoBERTa-BNE | configuración inicial | 5 | 0.40 | 0.0827 | 0.5641 | 0.5789 |

BETO también obtuvo F1=0.7407 en Confidentiality, 0.6154 en Integrity, 0.6667 en Authenticity y 0.6000 en Resistance. Non-repudiation y Accountability permanecieron en F1=0 para todos los modelos: las 5 y 6 semillas reales de train no aportan suficiente diversidad semántica aunque se generen variantes hasta balancear el conteo.

RoBERTa-BNE no aprende subcategorías en ninguno de los tres trials. Oscila entre todo negativo y la sobrepredicción de Security. El balance descarta al desbalance como única explicación; el checkpoint recuperado mediante mirror y su optimización son una limitación específica.

Estado: **completado**. Test continúa bloqueado y no fue utilizado.

## Revisión posterior — calidad semántica antes de nuevos ajustes

La mejora del experimento 2 no valida las subcategorías raras. Igualar sus conteos mediante variantes de las mismas 5–12 semillas repite los mismos patrones y no crea diversidad semántica. Además, la auditoría de las 19 filas reales marcadas como `Non_repudiation` o `Accountability` encontró varios posibles falsos positivos producidos por palabras como *disputa*, *registro* o *registrado*.

Medidas aplicadas:

1. Se generó `dataset_seguridad_v2_rare_labels_expert_review.csv` con texto, etiqueta actual, sugerencia razonada y columnas vacías para decisión experta. Las sugerencias son hipótesis de auditoría: no se cambió automáticamente el ground truth.
2. Se congelaron las 135 filas de test en `artifacts/security_test_frozen_seed42.csv`. SHA-256: `212ea80411b745d83f970c13e5c640bcddc777543fc81f13ed23e3b605e6998a`.
3. El aumento queda desactivado por defecto (`USE_TRAIN_AUGMENTATION=False`) hasta validar las semillas y conseguir ejemplos reales con escenarios distintos.
4. Se incorporó BCE moderadamente ponderada como opción (`USE_MODERATE_POS_WEIGHT`), calculada solo con train como `min(sqrt(N_neg/N_pos), 10)`. Permanece desactivada por defecto para que su efecto se mida aisladamente después de corregir etiquetas.
5. La selección de umbrales ahora es por etiqueta y usa exclusivamente validation. Test sigue sin intervenir.

### Diagnóstico controlado de RoBERTa-BNE

Se intentó memorizar un subconjunto de 48 ejemplos reales de train, sin duplicados ni textos con etiquetas contradictorias, durante 50 épocas (`lr=5e-5`, `max_len=64`, BCE ponderada). El mejor macro-F1 sobre ese mismo train fue **0.5332** y la mejor exactitud exacta multilabel fue **0.2292**; al final obtuvo macro-F1 **0.4944** y exactitud **0.0208**.

Conclusión: este checkpoint/mirror y su configuración actual no superan una prueba básica de sobreajuste. Por tanto, los resultados bajos de RoBERTa no deben atribuirse solo al desbalance. No se continuará ajustando sus hiperparámetros generales hasta resolver o sustituir la fuente del modelo. El diagnóstico completo se conserva en `artifacts/roberta_bne_overfit_diagnostic.csv` y `.json` y no modifica ningún checkpoint principal.

Estado: **pendiente de validación experta**. Los resultados del experimento 2 quedan como provisionales y no deben reportarse como resultado final. No se ejecutó test.

Limpieza: se eliminaron los checkpoints, predicciones, curvas, historiales y CSV de resultados del experimento 2 porque fueron entrenados con el aumento ahora descartado. La tabla anterior se conserva solo como trazabilidad histórica. Permanecen el manifiesto del split, la copia congelada de test y el diagnóstico aislado de RoBERTa.

