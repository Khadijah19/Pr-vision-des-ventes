# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 15:25:38 2026
@author: HP
"""

import pandas as pd
import numpy as np
import gc

from utils.data_loader import (
    load_items_hf,
    load_stores_hf,
    load_transactions_hf,
    load_oil_hf,
    load_holidays_hf,
)

from pathlib import Path
from collections.abc import Mapping, Sequence

def _to_pure_python(obj):
    # Convertit récursivement Path -> str
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {k: _to_pure_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__(_to_pure_python(x) for x in obj)
    return obj

class FavoritaFeaturePipeline:
    """
    Pipeline Fit/Transform pour créer les features Favorita sans fuite d'information.

    - feature_gap_days = retard d'accès aux données de ventes (et transactions si on le souhaite)
      Exemple feature_gap_days=3 :
        sales_roll14(t) utilise t-4..t-17
        sales_lag28(t)  utilise t-31
        transactions_roll14(t) utilise t-4..t-17 (dans ce pipeline)
    - oil : on laisse le calcul standard (shift(1) rolling 14), sans gap additionnel.
    - IMPORTANT : on garde NaN pour sales_lag28 et sales_roll14 quand l'historique n'existe pas.
    """

    def __init__(self, data_dir, sales_history_days=120, feature_gap_days=0, verbose=True):
        # ✅ IMPORTANT : éviter de stocker Path dans l'objet picklé
        self.data_dir = str(data_dir)

        self.sales_history_days = int(sales_history_days)
        self.feature_gap_days = int(feature_gap_days)
        self.verbose = bool(verbose)

        # learned objects
        self.store_freq = None
        self.item_freq = None

        self.items_fe = None
        self.store_maps = {}

        self.trans_s = None
        self.last_roll_by_store = None

        self.oil_s = None

        self.holiday_nat_maps = {}
        self.holiday_override_maps = {}

        # sales lookup history
        self.sales_s = None
        self.sales_target_col = None
        self.sales_history_max_date = None

    # ✅ Bonus: sécurise la sérialisation si jamais un Path traîne ailleurs
    def __getstate__(self):
        state = self.__dict__.copy()
        return _to_pure_python(state)

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ---------------------------
    # Utils
    # ---------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _ensure_datetime(df, col="date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
        return df

    @staticmethod
    def _make_key_store_date(store_series, date_series):
        """clé numérique compacte store-date"""
        ymd = date_series.dt.year * 10000 + date_series.dt.month * 100 + date_series.dt.day
        return (store_series.astype("int32") * 1_000_000 + ymd.astype("int32")).astype("int64")

    @staticmethod
    def add_target(df, y_col="unit_sales"):
        """
        Optionnel : prépare la cible si elle n'existe pas.
        - unit_sales_clean = clip des négatives à 0
        - unit_sales_log   = log1p(unit_sales_clean)
        """
        df = df.copy()
        if "unit_sales_clean" not in df.columns:
            df["unit_sales_clean"] = (
                pd.to_numeric(df[y_col], errors="coerce")
                .fillna(0)
                .clip(lower=0)
                .astype("float32")
            )
        if "unit_sales_log" not in df.columns:
            df["unit_sales_log"] = np.log1p(df["unit_sales_clean"]).astype("float32")
        return df

    # ---------------------------
    # FIT
    # ---------------------------
    def fit(self, history_df):
        """
        history_df : dataframe historique (train_fit) contenant
        date/store/item + cible (unit_sales_clean ou unit_sales_log)
        """
        history_df = history_df.copy()
        history_df = self._ensure_datetime(history_df, "date")

        self._log("🔧 FIT...")

        # ---- détecter cible
        if "unit_sales_log" in history_df.columns:
            target_col = "unit_sales_log"
        elif "unit_sales_clean" in history_df.columns:
            target_col = "unit_sales_clean"
        else:
            raise ValueError(
                "Il faut une colonne cible : unit_sales_clean ou unit_sales_log (tu peux appeler add_target avant)."
            )

        self.sales_target_col = target_col

        # forcer types
        history_df["store_nbr"] = pd.to_numeric(history_df["store_nbr"], errors="coerce").astype("int16")
        history_df["item_nbr"] = pd.to_numeric(history_df["item_nbr"], errors="coerce").astype("int32")

        # ---- freq enc (appris sur history_df)
        self.store_freq = history_df["store_nbr"].value_counts(normalize=True)
        self.item_freq = history_df["item_nbr"].value_counts(normalize=True)

        # ✅ normaliser index pour un map stable (int)
        self.store_freq.index = self.store_freq.index.astype("int16")
        self.item_freq.index = self.item_freq.index.astype("int32")

        # ---- Sales lookup (limité sur sales_history_days)
        max_d = history_df["date"].max()
        min_keep = max_d - pd.Timedelta(days=self.sales_history_days)

        hs = history_df.loc[
            history_df["date"] >= min_keep, ["store_nbr", "item_nbr", "date", target_col]
        ].copy()

        hs["store_nbr"] = hs["store_nbr"].astype("int16")
        hs["item_nbr"] = hs["item_nbr"].astype("int32")
        hs[target_col] = pd.to_numeric(hs[target_col], errors="coerce").astype("float32")

        hs = hs.drop_duplicates(["store_nbr", "item_nbr", "date"], keep="last")
        self.sales_history_max_date = hs["date"].max()
        self.sales_s = hs.set_index(["store_nbr", "item_nbr", "date"])[target_col]

        del hs
        gc.collect()

        # ---- ITEMS
        items = load_items_hf()
        items["item_nbr"] = pd.to_numeric(items["item_nbr"], errors="coerce").astype("int32")
        items["family"] = items["family"].fillna("UNKNOWN").astype(str).str.strip()
        items["class"] = items["class"].fillna(-1).astype("int16").astype(str)
        items["perishable"] = items["perishable"].fillna(0).astype("int8")

        fam_freq = items["family"].value_counts(normalize=True)
        cls_freq = items["class"].value_counts(normalize=True)

        items["family_freq_items"] = items["family"].map(fam_freq).astype("float32")
        items["class_freq_items"] = items["class"].map(cls_freq).astype("float32")

        # ✅ vectorisé / stable
        pair_counts = items.groupby(["family", "class"]).size()
        family_counts = items.groupby("family").size()

        mi = pd.MultiIndex.from_frame(items[["family", "class"]])
        num = pair_counts.reindex(mi).fillna(0).to_numpy(dtype="float32")
        den = items["family"].map(family_counts).fillna(0).to_numpy(dtype="float32")
        items["class_freq_in_family"] = np.where(den > 0, num / den, 0.0).astype("float32")

        self.items_fe = (
            items[["item_nbr", "perishable", "family_freq_items", "class_freq_items", "class_freq_in_family"]]
            .drop_duplicates("item_nbr")
        )

        del items
        gc.collect()

        # ---- STORES (freq apprises sur les OBS history_df)
        stores = load_stores_hf()
        stores["store_nbr"] = pd.to_numeric(stores["store_nbr"], errors="coerce").astype("int16")
        stores["type"] = stores["type"].fillna("UNKNOWN").astype(str).str.strip()
        stores["state"] = stores["state"].fillna("UNKNOWN").astype(str).str.strip()
        stores["city"] = stores["city"].fillna("UNKNOWN").astype(str).str.strip()
        stores["cluster"] = stores["cluster"].fillna(-1).astype("int16")

        hist_loc = history_df[["store_nbr"]].merge(
            stores[["store_nbr", "type", "state", "city", "cluster"]],
            on="store_nbr",
            how="left",
        )

        type_freq = hist_loc["type"].value_counts(normalize=True)
        state_freq = hist_loc["state"].value_counts(normalize=True)
        city_freq = hist_loc["city"].value_counts(normalize=True)

        stores["type_freq"] = stores["type"].map(type_freq).fillna(0).astype("float32")
        stores["state_freq"] = stores["state"].map(state_freq).fillna(0).astype("float32")
        stores["city_freq"] = stores["city"].map(city_freq).fillna(0).astype("float32")

        pair_counts = hist_loc.groupby(["state", "city"]).size()
        state_counts = hist_loc.groupby("state").size()

        mi_sc = pd.MultiIndex.from_frame(stores[["state", "city"]])
        num = pair_counts.reindex(mi_sc).fillna(0).to_numpy(dtype="float32")
        den = stores["state"].map(state_counts).fillna(0).to_numpy(dtype="float32")
        stores["city_freq_in_state"] = np.where(den > 0, num / den, 0.0).astype("float32")

        st_idx = stores.set_index("store_nbr")
        self.store_maps = {
            "cluster": st_idx["cluster"].to_dict(),
            "type_freq": st_idx["type_freq"].to_dict(),
            "state_freq": st_idx["state_freq"].to_dict(),
            "city_freq": st_idx["city_freq"].to_dict(),
            "city_freq_in_state": st_idx["city_freq_in_state"].to_dict(),
        }

        del stores, hist_loc
        gc.collect()

        # ---- TRANSACTIONS roll14 (avec le même gap que ventes)
        tr = load_transactions_hf()
        tr["date"] = pd.to_datetime(tr["date"], errors="coerce").dt.normalize()
        tr["store_nbr"] = pd.to_numeric(tr["store_nbr"], errors="coerce").astype("int16")
        tr["transactions"] = pd.to_numeric(tr["transactions"], errors="coerce").fillna(0)

        tr = (
            tr.groupby(["store_nbr", "date"], as_index=False)
            .agg(transactions=("transactions", "sum"))
            .sort_values(["store_nbr", "date"])
        )

        G = self.feature_gap_days
        tr["transactions_roll14"] = (
            tr.groupby("store_nbr")["transactions"]
            .transform(lambda x: x.shift(1 + G).rolling(14, min_periods=1).mean())
        ).astype("float32")

        self.trans_s = tr.set_index(["store_nbr", "date"])["transactions_roll14"]
        self.last_roll_by_store = tr.groupby("store_nbr")["transactions_roll14"].last()

        del tr
        gc.collect()

        # ---- OIL roll14 (standard, sans gap additionnel)
        oil = load_oil_hf().sort_values("date")
        oil["date"] = pd.to_datetime(oil["date"], errors="coerce").dt.normalize()
        oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce").ffill()
        oil["oil_roll14"] = oil["dcoilwtico"].shift(1).rolling(14, min_periods=1).mean().astype("float32")
        self.oil_s = oil.set_index("date")["oil_roll14"]

        del oil
        gc.collect()

        # ---- HOLIDAYS (national + override store)
        hol = load_holidays_hf()
        hol["date"] = pd.to_datetime(hol["date"], errors="coerce").dt.normalize()

        # ✅ robust: si certaines colonnes n’existent pas, on les crée
        for col in ["type", "locale", "locale_name", "description", "transferred"]:
            if col not in hol.columns:
                hol[col] = np.nan

        for col in ["type", "locale", "locale_name", "description"]:
            hol[col] = hol[col].fillna("").astype(str).str.strip()

        hol["transferred"] = hol["transferred"].fillna(False)
        # transferred peut être bool/str/int → on le normalise
        if hol["transferred"].dtype == "O":
            hol["transferred"] = hol["transferred"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        hol["transferred"] = hol["transferred"].astype(int)

        hol["event_key"] = hol["description"] + " | " + hol["locale"] + " | " + hol["locale_name"]

        transfer_map = (
            hol.loc[hol["type"] == "Transfer", ["event_key", "date"]]
            .drop_duplicates("event_key")
            .set_index("event_key")["date"]
            .to_dict()
        )

        hol["observed_date"] = hol["date"]
        mask_moved = (hol["type"] == "Holiday") & (hol["transferred"] == 1)
        hol.loc[mask_moved, "observed_date"] = hol.loc[mask_moved, "event_key"].map(transfer_map)
        hol["observed_date"] = hol["observed_date"].fillna(hol["date"])

        hol["f_holiday"] = (hol["type"] == "Holiday").astype("int8")

        stores_loc = load_stores_hf()
        stores_loc["store_nbr"] = pd.to_numeric(stores_loc["store_nbr"], errors="coerce").astype("int16")
        stores_loc["city"] = stores_loc["city"].fillna("UNKNOWN").astype(str).str.strip()
        stores_loc["state"] = stores_loc["state"].fillna("UNKNOWN").astype(str).str.strip()
        store_loc = stores_loc[["store_nbr", "city", "state"]].drop_duplicates("store_nbr")

        national = (
            hol[hol["locale"] == "National"]
            .groupby("observed_date", as_index=False)
            .agg(nat_h=("f_holiday", lambda s: int(s.sum() > 0)))
            .rename(columns={"observed_date": "date"})
        )

        self.holiday_nat_maps = {"nat_h": dict(zip(national["date"], national["nat_h"]))}

        regional = hol[hol["locale"] == "Regional"].copy()
        regional["date"] = regional["observed_date"]
        regional = regional.drop(columns=["observed_date"]).merge(
            store_loc, left_on="locale_name", right_on="state", how="inner"
        )
        regional = (
            regional.groupby(["date", "store_nbr"], as_index=False)
            .agg(h=("f_holiday", lambda s: int(s.sum() > 0)))
        )

        local = hol[hol["locale"] == "Local"].copy()
        local["date"] = local["observed_date"]
        local = local.drop(columns=["observed_date"]).merge(
            store_loc, left_on="locale_name", right_on="city", how="inner"
        )
        local = (
            local.groupby(["date", "store_nbr"], as_index=False)
            .agg(h=("f_holiday", lambda s: int(s.sum() > 0)))
        )

        over = (
            pd.concat([regional, local], ignore_index=True)
            .groupby(["date", "store_nbr"], as_index=False)
            .max()
        )
        over["key"] = self._make_key_store_date(over["store_nbr"], over["date"])
        self.holiday_override_maps = {"h": dict(zip(over["key"], over["h"]))}

        del hol, stores_loc, store_loc, national, regional, local, over
        gc.collect()

        self._log("✅ FIT terminé.")
        return self

    # ---------------------------
    # TRANSFORM
    # ---------------------------
    def transform(self, df_new):
        df = df_new.copy()
        df = self._ensure_datetime(df, "date")

        # types
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").astype("int16")
        df["item_nbr"] = pd.to_numeric(df["item_nbr"], errors="coerce").astype("int32")

        # ---- date feats
        d = df["date"]
        df["year"] = d.dt.year.astype("int16")
        df["month"] = d.dt.month.astype("int8")
        df["day"] = d.dt.day.astype("int8")
        df["dow"] = d.dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")

        # ---- promo (robuste)
        if "onpromotion" in df.columns:
            promo = df["onpromotion"]
            if promo.dtype == "O":
                promo_bool = promo.astype(str).str.lower().isin(["true", "1", "t", "yes"])
                promo_bool = promo_bool.where(promo.notna(), False)
                promo = promo_bool
            df["onpromo"] = promo.fillna(False).astype("int8")
        else:
            df["onpromo"] = np.int8(0)

        # ---- freq enc
        df["store_freq"] = (
            pd.to_numeric(df["store_nbr"].map(self.store_freq), errors="coerce")
            .fillna(0)
            .astype("float32")
        )
        df["item_freq"] = (
            pd.to_numeric(df["item_nbr"].map(self.item_freq), errors="coerce")
            .fillna(0)
            .astype("float32")
        )

        # ---- items merge
        df = df.merge(self.items_fe, on="item_nbr", how="left")
        df["perishable"] = pd.to_numeric(df["perishable"], errors="coerce").fillna(0).astype("int8")
        for c in ["family_freq_items", "class_freq_items", "class_freq_in_family"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

        # ---- stores maps
        df["cluster"] = (
            pd.to_numeric(df["store_nbr"].map(self.store_maps["cluster"]), errors="coerce")
            .fillna(-1)
            .astype("int16")
        )
        for c in ["type_freq", "state_freq", "city_freq", "city_freq_in_state"]:
            df[c] = (
                pd.to_numeric(df["store_nbr"].map(self.store_maps[c]), errors="coerce")
                .fillna(0)
                .astype("float32")
            )

        # ---- transactions_roll14
        keys = pd.MultiIndex.from_frame(df[["store_nbr", "date"]])
        vals = self.trans_s.reindex(keys).to_numpy(dtype="float32")
        df["transactions_roll14"] = pd.Series(vals, index=df.index)

        last_vals = pd.to_numeric(df["store_nbr"].map(self.last_roll_by_store), errors="coerce")
        df["transactions_roll14"] = df["transactions_roll14"].fillna(last_vals).fillna(0).astype("float32")

        # ---- oil_roll14
        oil_vals = self.oil_s.reindex(df["date"]).to_numpy(dtype="float32")
        df["oil_roll14"] = pd.Series(oil_vals, index=df.index)
        fallback_oil = float(self.oil_s.tail(30).mean()) if self.oil_s is not None and len(self.oil_s) else 0.0
        df["oil_roll14"] = df["oil_roll14"].fillna(fallback_oil).astype("float32")

        # ---- holidays
        df["is_holiday_effective"] = df["date"].map(self.holiday_nat_maps["nat_h"]).fillna(0).astype("int8")
        k = self._make_key_store_date(df["store_nbr"], df["date"])
        h_over = pd.Series(k, index=df.index).map(self.holiday_override_maps["h"]).fillna(0).to_numpy()
        df["is_holiday_effective"] = np.maximum(df["is_holiday_effective"].to_numpy(), h_over).astype("int8")

        # ============================
        # SALES LAG28 + ROLL14 (NaN + gap)
        # ============================
        G = self.feature_gap_days

        # --- LAG 28 (décalé)
        L = 28
        lag_idx = pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "store_nbr": df["store_nbr"].astype("int16"),
                    "item_nbr": df["item_nbr"].astype("int32"),
                    "date": df["date"] - pd.Timedelta(days=L + G),
                }
            )
        )
        lag_vals = self.sales_s.reindex(lag_idx).to_numpy(dtype="float32")
        df[f"sales_lag{L}"] = pd.Series(lag_vals, index=df.index).astype("float32")  # ✅ garde NaN
        df[f"sales_lag{L}_avail"] = (~pd.isna(lag_vals)).astype("int8")

        # --- ROLL 14 (t-(G+1) .. t-(G+14))
        W = 14
        acc = np.zeros(len(df), dtype="float32")
        cnt = np.zeros(len(df), dtype="int16")

        for j in range(1, W + 1):
            r_idx = pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {
                        "store_nbr": df["store_nbr"].astype("int16"),
                        "item_nbr": df["item_nbr"].astype("int32"),
                        "date": df["date"] - pd.Timedelta(days=G + j),
                    }
                )
            )
            vals = self.sales_s.reindex(r_idx).to_numpy(dtype="float32")
            mask = ~pd.isna(vals)
            acc[mask] += vals[mask]
            cnt[mask] += 1

        roll = np.full(len(df), np.nan, dtype="float32")
        np.divide(acc, cnt, out=roll, where=cnt > 0)

        df[f"sales_roll{W}"] = pd.Series(roll, index=df.index).astype("float32")
        df[f"sales_roll{W}_cnt"] = cnt.astype("int16")
        df[f"sales_roll{W}_frac"] = (cnt / float(W)).astype("float32")

        return df
