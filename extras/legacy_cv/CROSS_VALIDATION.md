# Protocolo de Cross-Validation multietiqueta

El pipeline activo está en `security_cross_validation.py`. Los cuatro notebooks son entrypoints
del mismo código y `run_all_training.py` ejecuta los cuatro modelos secuencialmente.

## Particiones

- El antiguo `train + val` forma un pool de 764 originales.
- `MultilabelStratifiedKFold` divide ese pool en 5 folds con semilla 42.
- El test congelado mantiene sus 135 casos y el SHA-256 registrado.
- Cada original ocupa validation exactamente una vez. Los textos duplicados se agrupan.
- Si se activa augmentation, se crea después del split y solo desde el train del fold.

Los umbrales por etiqueta se eligen exclusivamente con las predicciones OOF del modelo/configuración
ganadora. El número de épocas del modelo final es la media redondeada de las mejores épocas de los
cinco folds. El modelo final se inicializa desde cero y se entrena con todo el pool, sin early stopping
ni acceso al test.

## Ejecución

```powershell
python security_cross_validation.py --model cnn
python security_cross_validation.py --model bilstm
python security_cross_validation.py --model beto
python security_cross_validation.py --model roberta_bne
```

Opciones comunes: `--hparam-search`, `--use-augmentation` y `--moderate-pos-weight`.

La evaluación final está deliberadamente separada. Debe ejecutarse una sola vez cuando la
configuración ya esté cerrada:

```powershell
python security_cross_validation.py --model cnn --final-test
```

Al iniciarse la evaluación se crea `artifacts/<modelo>_final_test_consumed.json`; el pipeline rechaza
un segundo intento accidental. Para ejecutar todo el protocolo sobre los cuatro modelos se usa:

```powershell
python run_all_training.py
# Solo después de cerrar todas las decisiones experimentales:
python run_all_training.py --final-test
```

## Artefactos

Por modelo se guardan historial, trials, métricas de cada fold, agregados por etiqueta, predicciones
OOF, búsqueda de thresholds y resumen JSON. `artifacts/model_comparison_cv.csv` reúne la tabla final.
Los manifiestos y distribuciones auditables son:

- `artifacts/security_cv_folds_seed42.csv`
- `artifacts/security_cv_fold_label_distribution.csv`
- `artifacts/security_split_manifest_seed42.csv`
- `artifacts/security_test_frozen_seed42.sha256`

Las pruebas rápidas, que no predicen sobre test, se ejecutan con:

```powershell
python -m unittest -v test_security_cv_protocol.py
python run_all_training.py --protocol-only
```
