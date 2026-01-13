# utils/data_loader.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import hashlib
import pandas as pd

# ======================================================
# GLOBAL DEFAULT (✅ on fixe last10 partout)
# ======================================================
DEFAULT_PARQUET_WEEKS = 10


# ======================================================
# Helpers paths
# ======================================================
def _resolve_path(path_or_dir, filename: str) -> Path:
    """
    Si path_or_dir est un dossier -> join filename
    Sinon -> on considère que path_or_dir est déjà un chemin de fichier.
    """
    p = Path(path_or_dir)
    return (p / filename) if p.is_dir() else p


def _parquet_name(weeks: int) -> str:
    return f"train_last{int(weeks)}w.parquet"


# ======================================================
# Loaders (CSV légers)
# ======================================================
def load_items(data_dir: str) -> pd.DataFrame:
    path = _resolve_path(data_dir, "items.csv")
    if not path.exists():
        raise FileNotFoundError(f"items.csv introuvable: {path}")

    df = pd.read_csv(path)
    if "item_nbr" in df.columns:
        df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").fillna(0).astype("int32")
    if "family" in df.columns:
        df["family"] = df["family"].fillna("UNKNOWN").astype(str).str.strip()
    return df


def load_stores(data_dir: str) -> pd.DataFrame:
    path = _resolve_path(data_dir, "stores.csv")
    if not path.exists():
        raise FileNotFoundError(f"stores.csv introuvable: {path}")

    df = pd.read_csv(path)
    if "store_nbr" in df.columns:
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    return df


def load_oil(data_dir: str) -> pd.DataFrame:
    path = _resolve_path(data_dir, "oil.csv")
    if not path.exists():
        raise FileNotFoundError(f"oil.csv introuvable: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if "dcoilwtico" in df.columns:
        df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce")
    return df


def load_transactions(data_dir: str) -> pd.DataFrame:
    path = _resolve_path(data_dir, "transactions.csv")
    if not path.exists():
        raise FileNotFoundError(f"transactions.csv introuvable: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if "store_nbr" in df.columns:
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    if "transactions" in df.columns:
        df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype("float32")
    return df


def load_holidays(data_dir: str) -> pd.DataFrame:
    path = _resolve_path(data_dir, "holidays_events.csv")
    if not path.exists():
        raise FileNotFoundError(f"holidays_events.csv introuvable: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    for c in ["type", "locale", "locale_name", "description"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    if "transferred" in df.columns:
        df["transferred"] = df["transferred"].fillna(False).astype(bool)

    return df


# ======================================================
# Build recent parquet (streaming)
# ======================================================
def build_recent_parquet(
    data_dir: str,
    weeks: int = DEFAULT_PARQUET_WEEKS,
    in_filename: str = "train.csv",
    out_filename: str | None = None,
    chunksize: int = 2_000_000,
    usecols=("date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"),
) -> Path:
    """
    Construit train_last{weeks}w.parquet à partir de train.csv, sans charger tout en RAM.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / in_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"{in_filename} introuvable: {csv_path}")

    weeks = int(weeks)
    if out_filename is None:
        out_filename = _parquet_name(weeks)
    out_path = data_dir / out_filename

    # 1) trouver la date max
    max_date = None
    for chunk in pd.read_csv(csv_path, usecols=["date"], chunksize=chunksize):
        d = pd.to_datetime(chunk["date"], errors="coerce")
        m = d.max()
        if max_date is None or (pd.notna(m) and m > max_date):
            max_date = m

    if max_date is None or pd.isna(max_date):
        raise ValueError("Impossible de déterminer la date max dans train.csv")

    max_date = pd.to_datetime(max_date).normalize()
    start_date = max_date - pd.Timedelta(weeks=weeks)

    # 2) écrire parquet en streaming via pyarrow ParquetWriter
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out_path.exists():
        out_path.unlink()

    writer = None
    kept = 0

    for chunk in pd.read_csv(csv_path, usecols=list(usecols), chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        chunk = chunk.dropna(subset=["date"])
        chunk = chunk.loc[chunk["date"] >= start_date]

        if chunk.empty:
            continue

        if "onpromotion" in chunk.columns:
            chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool)

        chunk["unit_sales"] = pd.to_numeric(chunk["unit_sales"], errors="coerce").fillna(0).astype("float32")
        chunk["store_nbr"] = pd.to_numeric(chunk["store_nbr"], errors="coerce").fillna(0).astype("int16")
        chunk["item_nbr"]  = pd.to_numeric(chunk["item_nbr"],  errors="coerce").fillna(0).astype("int32")

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        kept += len(chunk)

    if writer is not None:
        writer.close()

    if kept == 0:
        raise ValueError(
            f"Aucune ligne gardée: start_date={start_date.date()} max_date={max_date.date()} "
            f"(vérifie tes dates / ton train.csv)"
        )

    return out_path


# ======================================================
# Load recent train (auto-build si manquant)
# ======================================================
def load_train_recent(
    data_dir: str,
    weeks: int = DEFAULT_PARQUET_WEEKS,
    parquet_name: str | None = None
) -> pd.DataFrame:
    """
    Charge un parquet (par défaut train_last10w.parquet).
    Si absent -> le construit automatiquement depuis train.csv.
    Puis sous-filtre sur weeks semaines (<= taille du parquet).
    """
    data_dir = Path(data_dir)
    weeks = int(weeks)

    if parquet_name is None:
        parquet_name = _parquet_name(DEFAULT_PARQUET_WEEKS)  # ✅ default = last10

    path = data_dir / parquet_name

    # auto-build si manquant
    if not path.exists():
        # ✅ construit toujours avec DEFAULT_PARQUET_WEEKS si parquet_name=last10
        #    et sinon il construit selon le parquet demandé
        if parquet_name.startswith("train_last") and parquet_name.endswith("w.parquet"):
            # ex: train_last10w.parquet -> build weeks=10
            try:
                w = int(parquet_name.replace("train_last", "").replace("w.parquet", ""))
            except Exception:
                w = DEFAULT_PARQUET_WEEKS
        else:
            w = DEFAULT_PARQUET_WEEKS

        build_recent_parquet(data_dir=str(data_dir), weeks=w, out_filename=parquet_name)

    df = pd.read_parquet(path)

    if "date" not in df.columns:
        raise ValueError(f"Colonne 'date' absente dans {path.name}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    if "store_nbr" in df.columns:
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype("int16")
    if "item_nbr" in df.columns:
        df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").fillna(0).astype("int32")
    if "unit_sales" in df.columns:
        df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce").fillna(0).astype("float32")
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)

    # sous-fenêtre (utile si parquet=last10w mais app veut 8w/4w)
    max_date = df["date"].max()
    start_date = max_date - pd.Timedelta(weeks=weeks)
    df = df.loc[df["date"] >= start_date]

    return df


# ======================================================
# Signature data (pour savoir si les données ont changé)
# ======================================================
def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def data_signature(data_dir: str, weeks: int = DEFAULT_PARQUET_WEEKS) -> dict:
    """
    Retourne une signature basée sur hash/mtime/size des fichiers clés.
    ✅ Par défaut on signe train_last10w.parquet.
    """
    data_dir = Path(data_dir)
    files = [
        data_dir / _parquet_name(int(weeks)),
        data_dir / "items.csv",
        data_dir / "stores.csv",
        data_dir / "oil.csv",
        data_dir / "transactions.csv",
        data_dir / "holidays_events.csv",
    ]

    out = {"weeks": int(weeks), "files": []}

    for p in files:
        if not p.exists():
            out["files"].append({
                "name": p.name,
                "path": str(p).replace("\\", "/"),
                "missing": True,
            })
            continue

        out["files"].append({
            "name": p.name,
            "path": str(p).replace("\\", "/"),
            "size": int(p.stat().st_size),
            "mtime": float(p.stat().st_mtime),
            "hash": _sha256_file(p),
            "hash_method": "sha256",
        })

    return out


# ======================================================
# Metadata models (json)
# ======================================================
def load_metadata(models_dir: str | Path = "models", filename: str = "metadata.json") -> dict:
    p = Path(models_dir) / filename
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def signatures_equal(sig_a, sig_b) -> bool:
    """
    Compare deux signatures (dict / json) de façon robuste.
    Retourne True si elles sont identiques, False sinon.
    """
    if sig_a is None or sig_b is None:
        return False

    try:
        import json
        return json.dumps(sig_a, sort_keys=True) == json.dumps(sig_b, sort_keys=True)
    except Exception:
        return sig_a == sig_b
