# utils/training.py
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline as SkPipeline

from utils.favorita_pipeline import FavoritaFeaturePipeline


# ======================================================
# Helpers
# ======================================================
def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _safe_expm1(x):
    # expm1 peut overflow si valeurs énormes => on clip un peu
    x = np.asarray(x, dtype="float64")
    x = np.clip(x, -50, 50)
    return np.expm1(x)

def split_84_gap_test(
    df: pd.DataFrame,
    total_days: int = 84,
    test_days: int = 14,
    gap_days: int = 3,
    date_col: str = "date",
):
    """
    Split temporel:
      - on garde les TOTAL_DAYS derniers jours
      - TEST = derniers TEST_DAYS
      - GAP = GAP_DAYS juste avant le test
      - TRAIN_FIT = le reste
    """
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
    """
    Sélection robuste des features:
    - exclut colonnes non-features
    - garde seulement numériques
    """
    drop_cols = {
        "date",
        "unit_sales",
        "unit_sales_clean",
        "unit_sales_log",
        "onpromotion",   # raw
    }

    # Garde uniquement numeric (évite les object)
    X_num = X_enriched.select_dtypes(include=[np.number]).copy()

    feature_cols = [c for c in X_num.columns if c not in drop_cols]
    feature_cols = sorted(feature_cols)

    return feature_cols


# ======================================================
# Main training
# ======================================================
def train_reference_model(
    df_last10w: pd.DataFrame,
    data_dir: str,
    models_dir: str = "models",
    weeks_window: int = 10,            # ✅ NEW
    feature_gap_days: int = 3,
    total_days: int = 84,
    test_days: int = 14,
    gap_days: int = 3,
    sales_history_days: int = 120,
    random_state: int = 42,
    data_signature: dict | None = None,
):
    """
    Entraîne le modèle "officiel" sur df_last10w (déjà filtré sur la fenêtre weeks_window),
    avec split 84j / gap / test.

    Sauvegarde:
      - models/best_model.pkl       (Sklearn pipeline = scaler + Ridge)
      - models/feature_pipeline.pkl (FavoritaFeaturePipeline fitted)
      - models/features.json
      - models/metadata.json
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    weeks_window = int(weeks_window)

    # 0) Cible
    df = df_last10w.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df = FavoritaFeaturePipeline.add_target(df, y_col="unit_sales")

    # 1) Split temporel officiel
    train_fit, gap_df, test_df, split_info = split_84_gap_test(
        df,
        total_days=total_days,
        test_days=test_days,
        gap_days=gap_days,
        date_col="date",
    )

    if len(train_fit) == 0 or len(test_df) == 0:
        raise ValueError(
            "Split impossible (train_fit ou test_df vide). "
            f"Vérifie que ta fenêtre ({weeks_window} semaines) contient bien >= {total_days} jours."
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
    X_test_full  = pipe.transform(test_df)

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
    y_test  = test_df["unit_sales_log"].astype("float32").to_numpy()

    # 5) Modèle stable (Scaler + Ridge)
    model = SkPipeline(steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=random_state)),
    ])
    model.fit(X_train, y_train)

    # 6) Évaluation (log + raw)
    pred_log = model.predict(X_test)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "StandardScaler + Ridge (log1p target)",

        # ✅ cohérence dashboard
        "weeks_window": weeks_window,

        **split_info,
        "n_features": int(len(feature_cols)),

        # log metrics
        "RMSE_log": _rmse(y_test, pred_log),
        "MAE_log": float(mean_absolute_error(y_test, pred_log)),
        "R2_log": float(r2_score(y_test, pred_log)),
    }

    # raw metrics (unit_sales)
    y_test_raw  = _safe_expm1(y_test)
    pred_raw    = _safe_expm1(pred_log)

    metrics.update({
        "RMSE_raw": _rmse(y_test_raw, pred_raw),
        "MAE_raw": float(mean_absolute_error(y_test_raw, pred_raw)),
        "R2_raw": float(r2_score(y_test_raw, pred_raw)),
        "Mean_y_true_raw": float(np.mean(y_test_raw)),
        "Mean_y_pred_raw": float(np.mean(pred_raw)),
    })

    # ✅ toujours mettre la signature si fournie
    if data_signature is not None:
        metrics["data_signature"] = data_signature

    # 7) Save artifacts
    joblib.dump(model, models_dir / "best_model.pkl")
    joblib.dump(pipe,  models_dir / "feature_pipeline.pkl")

    with open(models_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics
