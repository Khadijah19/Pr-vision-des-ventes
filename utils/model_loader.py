# utils/model_loader.py
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import joblib


# ======================================================
# PATHS (noms cohérents avec ton dashboard actuel)
# ======================================================
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH    = MODELS_DIR / "best_model.pkl"
PIPE_PATH     = MODELS_DIR / "feature_pipeline.pkl"
FEATURES_PATH = MODELS_DIR / "features.json"

# tes scripts récents écrivent metadata.json
METADATA_PATH = MODELS_DIR / "metadata.json"

# fallback si tu as encore l'ancien fichier
METRICS_PATH  = MODELS_DIR / "metrics.json"


# ======================================================
# LOAD METRICS
# ======================================================
def load_metrics(path: Path = None) -> dict | None:
    """
    Retourne un dict de métriques.
    Priorité: metadata.json, sinon metrics.json.
    """
    if path is None:
        path = METADATA_PATH if METADATA_PATH.exists() else METRICS_PATH

    path = Path(path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================
# LOAD ARTIFACTS (utilisé dans pages)
# ======================================================
def load_artifacts(
    model_path: Path = MODEL_PATH,
    pipe_path: Path = PIPE_PATH,
    features_path: Path = FEATURES_PATH,
):
    """
    Charge (model, pipe, feature_cols).
    Ne dépend d'aucun package externe (LightGBM etc.).
    """
    model_path = Path(model_path)
    pipe_path = Path(pipe_path)
    features_path = Path(features_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    if not pipe_path.exists():
        raise FileNotFoundError(f"Pipeline introuvable: {pipe_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features introuvables: {features_path}")

    model = joblib.load(model_path)
    pipe = joblib.load(pipe_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, pipe, feature_cols


# ======================================================
# BACKWARD COMPAT (si tu importais load_model_artifacts)
# ======================================================

def load_model_artifacts():
    """
    Alias pour compatibilité avec l'ancien code.
    """
    return load_artifacts()


# ======================================================
# Optional: helpers (si tu veux vérifier existence)
# ======================================================
def artifacts_exist() -> bool:
    return MODEL_PATH.exists() and PIPE_PATH.exists() and FEATURES_PATH.exists()
