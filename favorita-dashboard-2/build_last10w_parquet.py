# -*- coding: utf-8 -*-
"""
Build train_last10w.parquet from Favorita train.csv using only last 10 weeks.
ABSOLUTE PATHS (Windows) - reproducible in Spyder.

@author: HP
"""

from pathlib import Path
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# 1) ABSOLUTE PROJECT PATH (edit if needed)
# -----------------------------
PROJECT_DIR = Path(r"C:\Users\HP\Downloads\favorita-dashboard")  # <-- adapte si besoin
DATA_DIR = PROJECT_DIR / "data" / "favorita_data"

# input: train.csv OR train.csv.gz
csv_path = DATA_DIR / "train.csv"
if not csv_path.exists():
    gz_path = DATA_DIR / "train.csv.gz"
    if gz_path.exists():
        csv_path = gz_path
    else:
        raise FileNotFoundError(
            f"train.csv (ou train.csv.gz) introuvable dans {DATA_DIR}\n"
            f"Fichiers présents: {[p.name for p in DATA_DIR.glob('*')]}"
        )

# output
out_path = DATA_DIR / "train_last10w.parquet"

print("✅ PROJECT_DIR =", PROJECT_DIR)
print("📄 Train file  =", csv_path)
print("🧾 Output     =", out_path)


# -----------------------------
# 2) Build last 10 weeks parquet
# -----------------------------
usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
chunksize = 2_000_000

# 2.1 Find max date (scan only date)
max_date = None
for chunk in pd.read_csv(csv_path, usecols=["date"], chunksize=chunksize):
    d = pd.to_datetime(chunk["date"], errors="coerce")
    m = d.max()
    if max_date is None or (pd.notna(m) and m > max_date):
        max_date = m

if max_date is None or pd.isna(max_date):
    raise ValueError("Impossible de déterminer la date max dans train.csv")

start_date = max_date - pd.Timedelta(weeks=10)
print("📅 max_date =", max_date.date(), "| start_date =", start_date.date())

# 2.2 Stream write parquet for rows >= start_date
writer = None
kept_rows = 0

# overwrite existing output
if out_path.exists():
    out_path.unlink()

for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk = chunk.loc[chunk["date"] >= start_date]
    if chunk.empty:
        continue

    # cleaning / dtypes
    chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool)
    chunk["unit_sales"] = pd.to_numeric(chunk["unit_sales"], errors="coerce").fillna(0).astype("float32")
    chunk["store_nbr"] = chunk["store_nbr"].astype("int16")
    chunk["item_nbr"] = chunk["item_nbr"].astype("int32")

    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")

    writer.write_table(table)
    kept_rows += len(chunk)

if writer is not None:
    writer.close()

print(f"✅ écrit: {out_path} | lignes gardées: {kept_rows:,}")
