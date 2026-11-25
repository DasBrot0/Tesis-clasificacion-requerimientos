"""Configuración del proyecto: rutas, tareas, modelos e hiperparámetros.

Todo lo que se puede ajustar del experimento está en este archivo.

Diseño (replanteo del 24-11-2026, pedido del asesor): dos entrenamientos separados, con los mismos
cinco modelos y la misma validación cruzada anidada.

    Tarea 1, "binary":        seguridad vs. no seguridad, sobre todos los requisitos (1 salida).
    Tarea 2, "subcategories": solo los requisitos de seguridad, multietiqueta sobre las cuatro
                              subcategorías de ISO/IEC 25010 (4 salidas).

Condiciones:
    c0_real:      solo requisitos reales.
    c1_augmented: reales + sintéticos generados con LLM (solo en las particiones de entrenamiento).
                  Solo existe en la tarea 2: la tarea 1 se entrena siempre con datos reales.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- rutas

ROOT = Path(__file__).resolve().parent.parent

INPUTS = ROOT / "inputs"
ORIGINAL_DIR = INPUTS / "original"          # archivos fuente, sin modificar
INTERIM_DIR = INPUTS / "interim"            # glosario, traducciones, etiquetas y aumento
TRAINING_DIR = INPUTS / "training"          # lo que leen los modelos
AUGMENTATION_DIR = INTERIM_DIR / "augmentation"

OUTPUTS = ROOT / "outputs"
OUTPUT_RAW = OUTPUTS / "output_raw"         # CSV/JSON de cada corrida
REPORTS = OUTPUTS / "reports"               # Excel y figuras
LOGS = OUTPUTS / "logs"
CHECKPOINTS = OUTPUTS / "checkpoints"       # solo con --save-checkpoints (no se versiona)
SPLITS_DIR = OUTPUT_RAW / "splits"          # folds, compartidos por las dos tareas

# --------------------------------------------------------------------------- datos y etiquetas

SEED = 42
TEXT_COL = "Requirement_ES"
SUBCATEGORY_COLS = ["Confidentiality", "Integrity", "Authenticity", "Resistance"]
LABEL_COLS = ["Security", *SUBCATEGORY_COLS]  # columnas de la tabla completa de cada dataset

DATASETS = ["promise", "secreq", "dosspre"]
DATASET_NAMES = {"promise": "PROMISE_exp", "secreq": "SecReq", "dosspre": "DOSSPRE"}


def full_table_path(dataset: str) -> Path:
    """Tabla completa (Security + subcategorías). De ella salen los archivos de las dos tareas."""
    return TRAINING_DIR / f"{dataset}.csv"


def task_file_path(dataset: str, task: str) -> Path:
    """<dataset>_binary.csv (todos los requisitos) o <dataset>_subcategories.csv (solo seguridad)."""
    return TRAINING_DIR / f"{dataset}_{task}.csv"


def synthetic_path(dataset: str) -> Path:
    return AUGMENTATION_DIR / f"synthetic_{dataset}.csv"


# Una etiqueta con menos positivos que esto se predice, pero no entra en los promedios.
MIN_POSITIVES_PER_LABEL = 10

# --------------------------------------------------------------------------- tareas


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    metrics: tuple[str, ...]   # métricas que se reportan, en orden
    objective: str             # lo que maximiza Optuna y ordena Scott-Knott ESD
    tiebreak: str              # desempate entre épocas, trials y modelos
    conditions: tuple[str, ...]


TASKS = {
    # F1 de la clase seguridad como métrica principal (Petrillo et al., 2026).
    "binary": Task(
        name="binary",
        labels=("Security",),
        metrics=("f1", "precision", "recall", "f1_macro_2class"),
        objective="f1",
        tiebreak="recall",
        conditions=("c0_real",),
    ),
    "subcategories": Task(
        name="subcategories",
        labels=tuple(SUBCATEGORY_COLS),
        metrics=("macro_f1", "weighted_f1", "micro_f1", "macro_recall"),
        objective="macro_f1",
        tiebreak="macro_recall",
        conditions=("c0_real", "c1_augmented"),
    ),
}
CONDITIONS = ["c0_real", "c1_augmented"]


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Tarea desconocida: {name} (opciones: {list(TASKS)})")
    return TASKS[name]


# --------------------------------------------------------------------------- validación cruzada

# CV anidada: 5 folds externos estiman el rendimiento (cada requisito se prueba una vez) y, dentro
# del entrenamiento de cada externo, Optuna ajusta los hiperparámetros con 3 folds internos.
# Los folds externos se calculan una sola vez por dataset, sobre la tabla completa, y los usan las
# dos tareas: en el fold k, el modelo de subcategorías no vio ningún requisito que el modelo
# binario evalúa en ese fold, así que la cascada (src/cascade.py) no tiene fuga. [Decisión propia]
N_OUTER = 5
N_INNER = 3

# Trials de Optuna por fold externo (el trial 0 es siempre la configuración por defecto).
N_TRIALS = {"cnn": 25, "bilstm": 25, "beto": 12, "roberta_bne": 12, "distilbeto": 12}

# --------------------------------------------------------------------------- modelos

MODELS = ["cnn", "bilstm", "beto", "roberta_bne", "distilbeto"]
MODEL_DISPLAY = {"cnn": "CNN", "bilstm": "BiLSTM", "beto": "BETO", "roberta_bne": "RoBERTa-BNE",
                 "distilbeto": "DistilBETO"}
MODEL_FAMILY = {"cnn": "Deep Learning", "bilstm": "Deep Learning", "beto": "Transformers",
                "roberta_bne": "Transformers", "distilbeto": "Transformers"}
WORD_MODELS = {"cnn", "bilstm"}  # vocabulario de palabras propio; los demás usan su tokenizador

HF_SOURCES = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "roberta_bne": "PlanTL-GOB-ES/roberta-base-bne",
    "distilbeto": "dccuchile/distilbert-base-spanish-uncased",
}
# El repositorio oficial de RoBERTa-BNE ya no publica los pesos; este espejo tiene el mismo
# checkpoint, fijado en el commit que reproduce los puntajes oficiales.
ROBERTA_MIRROR = "Buzzeitor/roberta-base-bne"
HF_REVISIONS = {ROBERTA_MIRROR: "ef2fa9120bd771fd420893123a53ac9bfd8f0687"}
MAX_LENGTH = 128  # tokens de subpalabra; solo 4 requisitos de SecReq lo superan

# --------------------------------------------------------------------------- épocas y parada temprana

# Máximo de épocas y paciencia por tarea y modelo (REVISION_EPOCAS_PERDIDA_2026-11-24.md, repo de
# docs). Un fold se detiene cuando pasan `patience` épocas sin que mejore ni el objetivo de
# validación ni la pérdida de validación. El modelo final de cada fold externo se entrena con la
# media de las mejores épocas de sus folds internos.
# Topes de los Transformers: Kaur y Kaur (2023); Alqurashi y Alawneh (2025). Las paciencias de CNN y
# BiLSTM y los topes de la tarea de subcategorías son [Decisión propia], según las curvas previas.
MAX_EPOCHS = {
    "binary": {"cnn": 50, "bilstm": 50, "beto": 15, "roberta_bne": 15, "distilbeto": 15},
    "subcategories": {"cnn": 50, "bilstm": 50, "beto": 30, "roberta_bne": 30, "distilbeto": 30},
}
PATIENCE = {
    "binary": {"cnn": 10, "bilstm": 8, "beto": 3, "roberta_bne": 3, "distilbeto": 3},
    "subcategories": {"cnn": 10, "bilstm": 8, "beto": 5, "roberta_bne": 5, "distilbeto": 5},
}

# --------------------------------------------------------------------------- hiperparámetros

# Configuración por defecto de cada modelo (la de los notebooks originales). Optuna la evalúa
# siempre como trial 0, así que el modelo ajustado nunca queda peor que ella en la CV interna.
_TRANSFORMER_DEFAULT = {
    "learning_rate": 2e-5, "batch_size": 16, "dropout": 0.1, "weight_decay": 0.01,
    "optimizer": "AdamW", "warmup_ratio": 0.0, "pos_weight": False,
}
DEFAULT_CONFIGS = {
    "cnn": {"embedding_dim": 64, "num_filters": 100, "kernel_size": 3, "batch_size": 32,
            "dropout": 0.2, "learning_rate": 0.001, "pos_weight": False},
    "bilstm": {"embedding_dim": 100, "hidden_units": 64, "num_layers": 1, "batch_size": 32,
               "dropout": 0.2, "learning_rate": 0.001, "pos_weight": False},
    "beto": _TRANSFORMER_DEFAULT,
    "roberta_bne": _TRANSFORMER_DEFAULT,
    "distilbeto": _TRANSFORMER_DEFAULT,
}

# Espacios de búsqueda de Optuna: los de los notebooks originales, sin las épocas (las fija la
# parada temprana). Formato: ("categorical", opciones) | ("float", min, max) | ("log_float", min, max)
_TRANSFORMER_SPACE = {
    "learning_rate": ("log_float", 1e-5, 5e-5),
    "batch_size": ("categorical", [8, 16]),
    "dropout": ("float", 0.1, 0.5),
    "weight_decay": ("log_float", 1e-8, 0.3),
    "optimizer": ("categorical", ["AdamW", "Adam"]),
    "warmup_ratio": ("categorical", [0.0, 0.06, 0.1]),
    "pos_weight": ("categorical", [False, True]),
}
SEARCH_SPACES = {
    "cnn": {
        "embedding_dim": ("categorical", [64, 100, 128]),
        "num_filters": ("categorical", [100, 128]),
        "kernel_size": ("categorical", [3, 4, 5]),
        "batch_size": ("categorical", [16, 32]),
        "dropout": ("categorical", [0.2, 0.3, 0.4, 0.5]),
        "learning_rate": ("log_float", 1e-4, 1e-2),
        "pos_weight": ("categorical", [False, True]),
    },
    "bilstm": {
        "embedding_dim": ("categorical", [64, 100, 128]),
        "hidden_units": ("categorical", [32, 64, 128]),
        "num_layers": ("categorical", [1, 2]),
        "batch_size": ("categorical", [16, 32, 50]),
        "dropout": ("categorical", [0.2, 0.3, 0.5]),
        "learning_rate": ("log_float", 5e-4, 2e-3),
        "pos_weight": ("categorical", [False, True]),
    },
    "beto": _TRANSFORMER_SPACE,
    "roberta_bne": _TRANSFORMER_SPACE,
    "distilbeto": _TRANSFORMER_SPACE,
}

# Umbral de decisión durante el ajuste. Después se elige uno por etiqueta, con las predicciones
# fuera de fold de la CV interna, entre estos candidatos.
DEFAULT_THRESHOLD = 0.50
THRESHOLD_CANDIDATES = (0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600)

# --------------------------------------------------------------------------- rutas de salida


def raw_root(task: str) -> Path:
    return OUTPUT_RAW / task


def reports_root(task: str) -> Path:
    return REPORTS / task


def run_dir(task: str, condition: str, dataset: str, model: str) -> Path:
    return OUTPUT_RAW / task / condition / dataset / model


def checkpoint_dir(task: str, condition: str, dataset: str, model: str, outer: int) -> Path:
    return CHECKPOINTS / task / condition / dataset / model / f"outer_{outer}"


def log_name(task: str, condition: str, dataset: str, model: str) -> str:
    return f"{task}_{condition}_{dataset}_{model}.log"

# Predicciones de la tarea 2 sobre los requisitos sin seguridad de cada fold de prueba (para la cascada).
NONSECURITY_CSV = "nonsecurity_predictions.csv"
