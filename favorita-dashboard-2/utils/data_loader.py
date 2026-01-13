import pandas as pd

HF_BASE = "https://huggingface.co/datasets/khadidia-77/favorita/resolve/main"

def hf_url(filename: str) -> str:
    return f"{HF_BASE}/{filename}"

def load_train_from_hf(weeks: int = 10, filename: str = "train_last10w.parquet") -> pd.DataFrame:
    df = pd.read_parquet(hf_url(filename))
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    df["store_nbr"] = df["store_nbr"].astype("int16")
    df["item_nbr"] = df["item_nbr"].astype("int32")
    df["unit_sales"] = df["unit_sales"].astype("float32")
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)

    max_date = df["date"].max()
    start_date = max_date - pd.Timedelta(weeks=int(weeks))
    return df.loc[df["date"] >= start_date].copy()

def load_items_hf(filename: str = "items.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename))
    df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").fillna(0).astype("int32")
    df["family"] = df["family"].fillna("UNKNOWN").astype(str).str.strip()
    return df

def load_stores_hf(filename: str = "stores.csv") -> pd.DataFrame:
    df = pd.read_csv(hf_url(filename))
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
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
