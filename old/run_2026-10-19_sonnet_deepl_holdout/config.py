"""Central configuration: paths, labels, default hyperparameters and Optuna search spaces."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = ROOT / "datasets" / "original"
INTERIM_DIR = ROOT / "datasets" / "interim"
TRAINING_DIR = ROOT / "datasets" / "training"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINTS = ROOT / "checkpoints"

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.15
TEXT_COL = "Requirement_ES"
LABEL_COLS = [
    "Security",
    "Confidentiality",
    "Integrity",
    "Non_repudiation",
    "Accountability",
    "Authenticity",
    "Resistance",
]
SUBCATEGORY_COLS = LABEL_COLS[1:]
SUBCATEGORY_DEFINITIONS = {
    "Confidentiality": "protecting information from unauthorized disclosure",
    "Integrity": "preventing unauthorized modification of data",
    "Non_repudiation": "ensuring actions and transactions cannot later be denied",
    "Accountability": "enabling actions to be traced to responsible entities",
    "Authenticity": "verifying the identities of users, systems, and services",
    "Resistance": "withstanding, responding to, and recovering from attacks or malicious actions",
}
# A label needs at least this many positives to be split into test + N_SPLITS folds and
# still be scored. Labels below it stay in the model output but leave the macro averages.
MIN_POSITIVES_PER_LABEL = 10

DATASETS = {
    "promise": TRAINING_DIR / "promise.csv",
    "secreq": TRAINING_DIR / "secreq.csv",
    "dosspre": TRAINING_DIR / "dosspre.csv",
}
DATASET_NAMES = {"promise": "PROMISE_exp", "secreq": "SecReq", "dosspre": "DOSSPRE"}
MODELS = ["cnn", "bilstm", "beto", "roberta_bne"]
MODEL_DISPLAY = {"cnn": "CNN", "bilstm": "BiLSTM", "beto": "BETO", "roberta_bne": "RoBERTa-BNE"}
WORD_MODELS = {"cnn", "bilstm"}

HF_SOURCES = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "roberta_bne": "PlanTL-GOB-ES/roberta-base-bne",
}
# The official BNE repository no longer publishes weights; this mirror holds the same checkpoint.
ROBERTA_MIRROR = "Buzzeitor/roberta-base-bne"
# Pinned commits: the mirror reproduces the official fill-mask scores at this revision.
HF_REVISIONS = {ROBERTA_MIRROR: "ef2fa9120bd771fd420893123a53ac9bfd8f0687"}

# Default hyperparameters (the INITIAL_CONFIG of each original notebook). They are always
# evaluated as Optuna trial 0, so the tuned configuration is never worse than the default in CV.
# pos_weight and warmup_ratio did not exist in the notebooks; False / 0.0 reproduce them.
DEFAULT_CONFIGS = {
    "cnn": {
        "embedding_dim": 64, "num_filters": 100, "kernel_size": 3, "batch_size": 32,
        "epochs": 50, "dropout": 0.2, "learning_rate": 0.001, "pos_weight": False,
    },
    "bilstm": {
        "embedding_dim": 100, "hidden_units": 64, "num_layers": 1, "batch_size": 32,
        "epochs": 30, "dropout": 0.2, "learning_rate": 0.001, "pos_weight": False,
    },
    "beto": {
        "learning_rate": 2e-5, "batch_size": 16, "epochs": 5, "dropout": 0.1,
        "weight_decay": 0.01, "optimizer": "AdamW", "warmup_ratio": 0.0, "pos_weight": False,
    },
    "roberta_bne": {
        "learning_rate": 2e-5, "batch_size": 16, "epochs": 5, "dropout": 0.1,
        "weight_decay": 0.01, "optimizer": "AdamW", "warmup_ratio": 0.0, "pos_weight": False,
    },
}
# Not tuned: only 2 of the 2178 requirements exceed 128 subword tokens (the longest has 152).
MAX_LENGTH = 128
# Decision threshold while tuning; the per-label thresholds are chosen afterwards from OOF.
DEFAULT_THRESHOLD = 0.50
# BETO/RoBERTa-BNE stop a fold after this many epochs in which neither the validation macro-F1
# nor the validation loss improves, so "epochs" acts as an upper bound. CNN and BiLSTM always
# run their full number of epochs.
EARLY_STOPPING_PATIENCE = 3

# Optuna search spaces, taken from the SEARCH_SPACE declared in each original notebook.
# ("categorical", choices) | ("float", low, high) | ("log_float", low, high)
_TRANSFORMER_SPACE = {
    "learning_rate": ("log_float", 1e-5, 5e-5),
    "batch_size": ("categorical", [8, 16]),
    "epochs": ("categorical", [5, 10, 15, 20]),
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
        "epochs": ("categorical", [50, 60]),
        "dropout": ("categorical", [0.2, 0.3, 0.4, 0.5]),
        "learning_rate": ("log_float", 1e-4, 1e-2),
        "pos_weight": ("categorical", [False, True]),
    },
    "bilstm": {
        "embedding_dim": ("categorical", [64, 100, 128]),
        "hidden_units": ("categorical", [32, 64, 128]),
        "num_layers": ("categorical", [1, 2]),
        "batch_size": ("categorical", [16, 32, 50]),
        "epochs": ("categorical", [20, 30, 50]),
        "dropout": ("categorical", [0.2, 0.3, 0.5]),
        "learning_rate": ("log_float", 5e-4, 2e-3),
        "pos_weight": ("categorical", [False, True]),
    },
    "beto": _TRANSFORMER_SPACE,
    "roberta_bne": _TRANSFORMER_SPACE,
}
N_TRIALS = {"cnn": 30, "bilstm": 30, "beto": 12, "roberta_bne": 12}


def run_dir(dataset: str, model: str) -> Path:
    return ARTIFACTS / dataset / model


def checkpoint_dir(dataset: str, model: str) -> Path:
    return CHECKPOINTS / dataset / model / "final"
