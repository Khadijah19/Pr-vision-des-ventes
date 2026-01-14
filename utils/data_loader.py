# utils/data_loader.py
# -*- coding: utf-8 -*-

import pandas as pd

HF_BASE = "https://huggingface.co/datasets/khadidia-77/favorita/resolve/main"

def hf_url(filename: str) -> str:
    return f"{HF_BASE}/{filename}"

# ----------------------------
# HF loaders (source of truth)
# ----------------------------
from huggingface_hub import hf_hub_download

HF_DATASET_REPO = "khadidia-77/favorita"
HF_REPO_TYPE = "dataset"

def load_train_from_hf(
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
    weeks: int = 10,
    filename: str = "train_last10w.parquet",
    cache_dir: str = ".cache/favorita_data",
) -> pd.DataFrame:
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        filename=filename,
        cache_dir=cache_dir,
        token=hf_token,
    )

    df = pd.read_parquet(local)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    df["store_nbr"] = df["store_nbr"].astype("int16")
    df["item_nbr"] = df["item_nbr"].astype("int32")
    if "unit_sales" in df.columns:
        df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce").fillna(0).astype("float32")
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)

    max_date = df["date"].max()
    start_date = max_date - pd.Timedelta(weeks=int(weeks))
    return df.loc[df["date"] >= start_date].copy()


def load_items_hf(filename: str = "items.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename))
    df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").fillna(0).astype("int32")
    df["family"] = df["family"].fillna("UNKNOWN").astype(str).str.strip()
    if "class" in df.columns:
        df["class"] = df["class"].fillna(-1)
    if "perishable" in df.columns:
        df["perishable"] = pd.to_numeric(df["perishable"], errors="coerce").fillna(0).astype("int8")
    return df

def load_stores_hf(filename: str = "stores.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename))
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    for c in ["city", "state", "type"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype("int16")
    return df

def load_oil_hf(filename: str = "oil.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename), parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce")
    return df

def load_transactions_hf(filename: str = "transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename), parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype("float32")
    return df

def load_holidays_hf(filename: str = "holidays_events.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename), parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ["type", "locale", "locale_name", "description"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    if "transferred" in df.columns:
        df["transferred"] = df["transferred"].fillna(False).astype(bool)
    return df

# -------------------------------------------
# COMPAT LAYER (pour tes pages existantes)
# -------------------------------------------
def load_train_recent(data_dir=None, weeks: int = 10, parquet_name: str = "train_last10w.parquet") -> pd.DataFrame:
    # data_dir ignoré (HF only)
    return load_train_from_hf(weeks=weeks, filename=parquet_name)

def load_items(data_dir=None) -> pd.DataFrame:
    return load_items_hf()

def load_stores(data_dir=None) -> pd.DataFrame:
    return load_stores_hf()

def load_oil(data_dir=None) -> pd.DataFrame:
    return load_oil_hf()

def load_transactions(data_dir=None) -> pd.DataFrame:
    return load_transactions_hf()

def load_holidays(data_dir=None) -> pd.DataFrame:
    return load_holidays_hf()



# -------------------------------------------
# ADMIN helpers (local artifacts only)
# -------------------------------------------
import json
from pathlib import Path

def load_metadata(models_dir: str | Path = "models") -> dict:
    """Lit models/metadata.json (ou renvoie {} si absent)."""
    p = Path(models_dir) / "metadata.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_metrics(models_dir: str | Path = "models") -> dict:
    """Alias: certains scripts appellent load_metrics."""
    return load_metadata(models_dir=models_dir)
