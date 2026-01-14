# utils/training.py
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.favorita_pipeline import FavoritaFeaturePipeline


# ======================================================
# Small file helpers
# ======================================================
def _write_json(path: Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _copytree_overwrite(src_dir: Path, dst_dir: Path):
    """Copie tout le contenu de src_dir vers dst_dir en écrasant les fichiers."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.glob("*"):
        target = dst_dir / p.name
        if p.is_dir():
            shutil.copytree(p, target, dirs_exist_ok=True)
        else:
            shutil.copy2(p, target)


# ======================================================
# Metrics helpers
# ======================================================
def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_expm1(x):
    x = np.asarray(x, dtype="float64")
    x = np.clip(x, -50, 50)
    return np.expm1(x)


# ======================================================
# Split helper (84 days with a gap)
# ======================================================
def split_84_gap_test(
    df: pd.DataFrame,
    total_days: int = 84,
    test_days: int = 14,
    gap_days: int = 3,
    date_col: str = "date",
):
    if date_col not in df.columns:
        raise ValueError(f"Colonne date introuvable: {date_col}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    max_date = df[date_col].max()
    start_total = max_date - pd.Timedelta(days=total_days - 1)
    df_total = df.loc[df[date_col] >= start_total].copy()

    # TEST
    end_test = max_date
    start_test = end_test - pd.Timedelta(days=test_days - 1)

    # GAP (avant test)
    end_gap = start_test - pd.Timedelta(days=1)
    start_gap = end_gap - pd.Timedelta(days=gap_days - 1)

    train_fit = df_total.loc[df_total[date_col] < start_gap].copy()
    gap_df = df_total.loc[(df_total[date_col] >= start_gap) & (df_total[date_col] <= end_gap)].copy()
    test_df = df_total.loc[(df_total[date_col] >= start_test) & (df_total[date_col] <= end_test)].copy()

    info = {
        "total_days": int(total_days),
        "test_days": int(test_days),
        "gap_days": int(gap_days),
        "train_min_date": str(train_fit[date_col].min().date()) if len(train_fit) else None,
        "train_max_date": str(train_fit[date_col].max().date()) if len(train_fit) else None,
        "gap_min_date": str(gap_df[date_col].min().date()) if len(gap_df) else None,
        "gap_max_date": str(gap_df[date_col].max().date()) if len(gap_df) else None,
        "test_min_date": str(test_df[date_col].min().date()) if len(test_df) else None,
        "test_max_date": str(test_df[date_col].max().date()) if len(test_df) else None,
        "n_train_fit": int(len(train_fit)),
        "n_gap": int(len(gap_df)),
        "n_test": int(len(test_df)),
        "max_date": str(max_date.date()) if pd.notna(max_date) else None,
        "start_total": str(start_total.date()) if pd.notna(start_total) else None,
    }

    return train_fit, gap_df, test_df, info


def select_feature_cols(X_enriched: pd.DataFrame):
    """On garde uniquement les colonnes numériques (et on retire la cible, date, etc.)."""
    drop_cols = {
        "date",
        "unit_sales",
        "unit_sales_clean",
        "unit_sales_log",
        "onpromotion",  # raw
    }
    X_num = X_enriched.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in X_num.columns if c not in drop_cols]
    return sorted(feature_cols)


# ======================================================
# HuggingFace cache helpers
# ======================================================
def ensure_hf_cache(
    repo_id: str,
    cache_dir: str = "data/hf_cache",
    force: bool = False,
    allow_patterns: list[str] | None = None,
    hf_token: str | None = None,
    repo_type: str = "dataset",  # <-- très important
) -> Path:
    """
    Télécharge (ou réutilise) un snapshot du repo HF en local (cache),
    et renvoie le chemin du snapshot.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError("Installe huggingface_hub: pip install huggingface_hub") from e

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,              # <-- dataset vs model
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
        force_download=bool(force),
        token=hf_token,                   # <-- utilise le token
    )
    return Path(snapshot_path)



def resolve_data_dir(snapshot_path: Path) -> Path:
    """
    Cherche le dossier qui contient les CSV attendus par FavoritaFeaturePipeline.
    On teste d'abord snapshot/data/favorita_data puis snapshot directement,
    sinon recherche récursive du dossier contenant items.csv.
    """
    snapshot_path = Path(snapshot_path)

    candidate = snapshot_path / "data" / "favorita_data"
    if (candidate / "items.csv").exists():
        return candidate

    if (snapshot_path / "items.csv").exists():
        return snapshot_path

    found = list(snapshot_path.rglob("items.csv"))
    if found:
        return found[0].parent

    raise FileNotFoundError(f"Impossible de trouver items.csv dans le snapshot HF: {snapshot_path}")


# ======================================================
# Main training (LightGBM)
# ======================================================
def train_reference_model(
    df_last10w: pd.DataFrame,
    data_dir: str,
    models_dir: str = "models",
    weeks_window: int = 10,
    feature_gap_days: int = 3,
    total_days: int = 84,
    test_days: int = 14,
    gap_days: int = 3,
    sales_history_days: int = 120,
    random_state: int = 42,
    data_signature: dict | None = None,
) -> dict:
    # Import LightGBM ici (pour message d'erreur clair si non installé)
    try:
        import lightgbm as lgb
    except Exception as e:
        raise ImportError("LightGBM n'est pas installé. Fais: pip install lightgbm") from e

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    weeks_window = int(weeks_window)

    # 0) Cible (log1p)
    df = df_last10w.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df = FavoritaFeaturePipeline.add_target(df, y_col="unit_sales")

    # 1) Split temporel
    train_fit, gap_df, test_df, split_info = split_84_gap_test(
        df,
        total_days=int(total_days),
        test_days=int(test_days),
        gap_days=int(gap_days),
        date_col="date",
    )

    if len(train_fit) == 0 or len(test_df) == 0:
        raise ValueError(
            "Split impossible (train_fit ou test_df vide). "
            f"Ta fenêtre ({weeks_window} semaines) doit contenir au moins {total_days} jours."
        )

    # 2) Fit Feature Pipeline sur TRAIN_FIT uniquement
    pipe = FavoritaFeaturePipeline(
        data_dir=str(data_dir),
        sales_history_days=int(sales_history_days),
        feature_gap_days=int(feature_gap_days),
        verbose=True,
    )
    pipe.fit(train_fit)

    # 3) Transform
    X_train_full = pipe.transform(train_fit)
    X_test_full = pipe.transform(test_df)

    # 4) Features + matrices
    feature_cols = select_feature_cols(X_train_full)

    X_train = (
        X_train_full.reindex(columns=feature_cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    X_test = (
        X_test_full.reindex(columns=feature_cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_train = train_fit["unit_sales_log"].astype("float32").to_numpy()
    y_test = test_df["unit_sales_log"].astype("float32").to_numpy()

    # 5) Modèle LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=int(random_state),
        n_jobs=-1,
    )

    model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    )


    # 6) Évaluation (log + raw)
    pred_log = model.predict(X_test)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "LightGBM LGBMRegressor (log1p target)",
        "weeks_window": int(weeks_window),
        **split_info,
        "n_features": int(len(feature_cols)),
        "RMSE_log": _rmse(y_test, pred_log),
        "MAE_log": float(mean_absolute_error(y_test, pred_log)),
        "R2_log": float(r2_score(y_test, pred_log)),
    }

    y_test_raw = _safe_expm1(y_test)
    pred_raw = _safe_expm1(pred_log)

    metrics.update(
        {
            "RMSE_raw": _rmse(y_test_raw, pred_raw),
            "MAE_raw": float(mean_absolute_error(y_test_raw, pred_raw)),
            "R2_raw": float(r2_score(y_test_raw, pred_raw)),
            "Mean_y_true_raw": float(np.mean(y_test_raw)),
            "Mean_y_pred_raw": float(np.mean(pred_raw)),
        }
    )

    if data_signature is not None:
        metrics["data_signature"] = data_signature

    # 7) Save artifacts (overwrite)
    joblib.dump(model, models_dir / "best_model.pkl")
    joblib.dump(pipe, models_dir / "feature_pipeline.pkl")

    with open(models_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


# ======================================================
# Wrapper "model_trainer style" (HF cache + train parquet HF)
# ======================================================
def train_and_save(
    repo_id: str,
    weeks_window: int = 10,
    parquet_name: str = "train_last10w.parquet",
    artifacts_root: str = "artifacts",
    models_dir: str = "models",
    hf_cache_dir: str = "data/hf_cache",
    # training params
    total_days: int = 84,
    test_days: int = 14,
    gap_days: int = 3,
    feature_gap_days: int = 3,
    sales_history_days: int = 120,
    random_state: int = 42,
    force_cache: bool = False,
) -> dict:
    """
    Entraîne + sauvegarde:
      artifacts/runs/<run_id>/{best_model.pkl, feature_pipeline.pkl, features.json, metadata.json}
      artifacts/latest.json  -> pointe vers le dernier run
      models/*               -> copie "courante" (écrasée à chaque entraînement)
    """
    # Import local (évite des soucis d'import au chargement)
    from utils.data_loader import load_train_from_hf

    # 1) Cache HF local pour les CSV (items/stores/transactions/oil/holidays)
    snapshot = ensure_hf_cache(
        repo_id=repo_id,
        cache_dir=hf_cache_dir,
        force=force_cache,
    )
    data_dir = resolve_data_dir(snapshot)

    # 2) Charger le train parquet depuis HF (fenêtre weeks)
    df = load_train_from_hf(repo_id=repo_id, weeks=int(weeks_window), filename=parquet_name)

    # 3) Run id + dossiers
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(artifacts_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4) Entraîner en sauvegardant dans run_dir
    metrics = train_reference_model(
        df_last10w=df,
        data_dir=str(data_dir),
        models_dir=str(run_dir),
        weeks_window=int(weeks_window),
        feature_gap_days=int(feature_gap_days),
        total_days=int(total_days),
        test_days=int(test_days),
        gap_days=int(gap_days),
        sales_history_days=int(sales_history_days),
        random_state=int(random_state),
        data_signature={
            "source": "huggingface",
            "repo_id": repo_id,
            "parquet": parquet_name,
            "weeks_window": int(weeks_window),
        },
    )

    # 5) latest.json -> pointe vers ce run
    latest = {
        "run_id": run_id,
        "run_dir": str(run_dir).replace("\\", "/"),
        "models_dir": str(models_dir).replace("\\", "/"),
        "trained_at": metrics.get("trained_at"),
        "weeks_window": int(weeks_window),
        "repo_id": repo_id,
        "parquet_name": parquet_name,
    }
    _write_json(Path(artifacts_root) / "latest.json", latest)

    # 6) Copier comme "modèles courants" (écrase models/*)
    cur_models = Path(models_dir)
    cur_models.mkdir(parents=True, exist_ok=True)
    _copytree_overwrite(run_dir, cur_models)

    # Optionnel: info cache
    _write_json(
        run_dir / "hf_cache_info.json",
        {
            "snapshot_dir": str(snapshot).replace("\\", "/"),
            "data_dir": str(data_dir).replace("\\", "/"),
        },
    )

    return {"latest": latest, "metrics": metrics}


# ======================================================
# Train + publish HF
# ======================================================
def train_and_publish(
    weeks_window: int = 10,
    parquet_name: str = "train_last10w.parquet",
    models_dir: str = "models",
    hf_repo_id: str = "khadidia-77/favorita",
    hf_token: str | None = None,
    hf_cache_dir: str = "data/hf_cache",
    force_cache: bool = False,
) -> dict:
    """
    1) cache HF (CSV) + resolve_data_dir
    2) charge df depuis HF (parquet)
    3) entraîne + écrit dans models/
    4) publie vers HF (runs/<timestamp>/...) + update latest.json
    """
    from utils.data_loader import load_train_from_hf
    from utils.hf_artifacts import publish_run_to_hf

    # 1) Cache CSV + data_dir
    snapshot = ensure_hf_cache(
    repo_id=hf_repo_id,
    cache_dir=hf_cache_dir,
    force=force_cache,
    hf_token=hf_token,
    repo_type="dataset",   # ou "model" selon ton cas
    )
    
    data_dir = resolve_data_dir(snapshot)

    # 2) Parquet train
    df = load_train_from_hf(weeks=int(weeks_window), filename=parquet_name)

    # 3) Train local models/
    metrics = train_reference_model(
        df_last10w=df,
        data_dir=str(data_dir),
        models_dir=models_dir,
        weeks_window=int(weeks_window),
        data_signature={
            "source": "huggingface",
            "repo_id": hf_repo_id,
            "parquet": parquet_name,
            "weeks_window": int(weeks_window),
        },
    )

    # 4) Publish HF
    latest = publish_run_to_hf(
        local_models_dir=models_dir,
        repo_id=hf_repo_id,
        hf_token=hf_token,
    )

    return {"train_metrics": metrics, "published": latest}
