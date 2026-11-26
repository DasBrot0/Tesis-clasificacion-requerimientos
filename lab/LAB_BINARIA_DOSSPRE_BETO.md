# Laboratorio: tarea binaria de BETO en DOSSPRE

Instrucciones para un agente Claude Code en una PC del laboratorio (RTX 6000 Ada de 48 GB). Hay
que entrenar **una sola corrida completa**: la tarea binaria (seguridad / no seguridad), condición
C0, dataset DOSSPRE, modelo BETO. Después hay que empaquetar los resultados para llevarlos a la
laptop del usuario.

El usuario tiene la PC de **11:00 a 13:00**. La instalación y la corrida tienen que caber en ese
tiempo.

## Reglas

- Corre solo esta corrida. No entrenes otros modelos, tareas ni datasets: la laptop del usuario
  se encarga del resto, y una corrida repetida en dos máquinas no sirve.
- La corrida se hace **completa en esta PC**: los 5 folds externos. Si no termina a tiempo, no
  se lleva a medias a otra máquina (ver «Si no alcanza el tiempo»).
- No modifiques el código, `inputs/` ni `outputs/output_raw/splits/`. Tampoco corras
  `scripts/prepare_datasets.py`, porque los datos de entrenamiento y los folds ya vienen en el repo.
- No hagas push ni commits. Los resultados se llevan en un `.zip`.
- Si la PC también corre la otra corrida del laboratorio (`lab/LAB_BINARIA_DOSSPRE_ROBERTA_BNE.md`), lanza las dos en
  paralelo, cada una en su terminal. Las dos entran en la GPU: cada una usa unos 3 GB.

## 1. Preparar el entorno (≈ 15-20 min)

Sirve cualquier Python entre 3.10 y 3.14. Comprueba también la GPU con `nvidia-smi`. La laptop
usa Python 3.14.5, PyTorch 2.14.0 (CUDA 12.6), Transformers 5.17.0, Optuna 5.0.0,
scikit-learn 1.9.1, pandas 3.0.3 y numpy 2.4.6. Versiones cercanas sirven; anota las que queden.

Windows (PowerShell):

```powershell
git clone --branch replanteo-binario-subcategorias https://github.com/DasBrot0/Tesis-clasificacion-requerimientos.git tesis
cd tesis
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Linux:

```bash
git clone --branch replanteo-binario-subcategorias https://github.com/DasBrot0/Tesis-clasificacion-requerimientos.git tesis
cd tesis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

- Si no hay `git`, descarga el `.zip` de la rama `replanteo-binario-subcategorias` desde GitHub.
- Si el usuario trae un USB con el repo o con la caché de Hugging Face (`~/.cache/huggingface`
  o `%USERPROFILE%\.cache\huggingface`), úsalos y te ahorras descargas.

Verifica que la GPU se vea y que el código funcione:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m unittest discover -s tests
```

Tiene que imprimir `True` y el nombre de la RTX 6000 Ada, y las pruebas tienen que pasar. Si
`cuda.is_available()` da `False`, reinstala torch con el índice `cu124` o `cu128` según el
controlador que muestre `nvidia-smi`. No entrenes en CPU.

Descarga el modelo antes de lanzar la corrida, así un error de red aparece ahora y no a mitad del
entrenamiento:

```bash
python -c "from src.models import transformer_source; from src.config import HF_REVISIONS; from transformers import AutoTokenizer, AutoModel; s = transformer_source('beto'); r = HF_REVISIONS.get(s); AutoTokenizer.from_pretrained(s, revision=r); AutoModel.from_pretrained(s, revision=r); print('ok', s)"
```

## 2. Lanzar la corrida (≈ 40-90 min, estimado)

Windows:

```powershell
$env:PYTHONIOENCODING = "utf-8"; $env:MPLBACKEND = "Agg"
python run_all_training.py --tasks binary --models beto --datasets dosspre
```

Linux:

```bash
PYTHONIOENCODING=utf-8 MPLBACKEND=Agg python run_all_training.py --tasks binary --models beto --datasets dosspre
```

El progreso detallado se escribe en `outputs/logs/binary_c0_real_dosspre_beto.log`. Cada
fold externo termina cuando aparece `outputs/output_raw/binary/c0_real/dosspre/beto/outer_<k>/outer_result.json`
(k = 0…4). La corrida terminó cuando la terminal imprime `FIN binary/c0_real/dosspre/beto`
seguido de `ENTRENAMIENTO COMPLETO`.

La duración no está medida en esta GPU. En la laptop del usuario (RTX 3050 de 4 GB) esta corrida
tomaría unas 3,2 h, y aquí se espera que vaya de 4 a 8 veces más rápido.

**Estimación temprana:** cuando aparezca `outer_0/outer_result.json`, multiplica el tiempo que tomó
por 5 y avísale al usuario la hora estimada de término.

## Si no alcanza el tiempo

Si la estimación indica que la corrida no termina antes de las **12:50**, avísale al usuario
apenas lo sepas. Si a las 12:50 la corrida no terminó:

- detenla (Ctrl+C) y no empaquetes nada;
- dile al usuario que la corrida queda para su laptop, donde se hará desde cero.

## 3. Verificar y empaquetar

Al terminar, comprueba que la corrida esté completa:

```bash
python -c "import json; d=json.load(open('outputs/output_raw/binary/c0_real/dosspre/beto/summary.json')); print(len(d['outer_folds']), 'folds;', round(d['elapsed_minutes'],1), 'min;', {k: round(v,3) for k,v in d['pooled_outer_test'].items()})"
```

Tiene que decir `5 folds`. Las métricas son el F1, la precisión y el recall de la clase seguridad.

Guarda los datos del entorno en un archivo de texto, porque la sección de Experimentación del
paper menciona el hardware y las versiones:

```bash
nvidia-smi > outputs/entorno_lab_beto.txt
python -c "import sys,platform,torch,transformers,optuna,sklearn,pandas,numpy; print(sys.version); print(platform.platform()); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'optuna', optuna.__version__, 'sklearn', sklearn.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__)" >> outputs/entorno_lab_beto.txt
```

Agrega también al final de ese archivo, a mano o con un comando, el procesador, la RAM y el sistema
operativo de la PC.

Empaqueta la carpeta de la corrida, su log y el archivo del entorno, con las rutas relativas a la
raíz del repo:

Windows:

```powershell
Compress-Archive -Path outputs\output_raw\binary\c0_real\dosspre\beto, outputs\logs\binary_c0_real_dosspre_beto.log, outputs\entorno_lab_beto.txt -DestinationPath lab_binary_dosspre_beto.zip
```

Linux:

```bash
zip -r lab_binary_dosspre_beto.zip outputs/output_raw/binary/c0_real/dosspre/beto outputs/logs/binary_c0_real_dosspre_beto.log outputs/entorno_lab_beto.txt
```

`Compress-Archive` guarda la carpeta sin la ruta `outputs\output_raw\...`. No es problema: en la
laptop se descomprime en su lugar. Copia el `.zip` al USB o a Google Drive del usuario y dile dónde
quedó.

## 4. Qué reportarle al usuario

- Hora de inicio y de término, y los minutos de `elapsed_minutes`.
- Las métricas agrupadas (F1, precisión, recall y macro-F1 de dos clases).
- El nombre y la ubicación del `.zip`.
- Cualquier advertencia o error del log.

En la laptop, el `.zip` se descomprime así:

- la carpeta `beto` va en `outputs/output_raw/binary/c0_real/dosspre/beto/`;
- el log va en `outputs/logs/`.

Después se corre `py scripts/build_report.py`. Eso lo hace el agente de la laptop, no el del
laboratorio.
