# pages/2_Prédictions.py
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np

# ✅ Fix import utils sur Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]   # repo root (où se trouve utils/)
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_train_from_hf
from utils.hf_artifacts import download_artifacts_from_latest


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Prédictions - Favorita",
    page_icon="🔮",
    layout="wide",
)

# --- HF settings (dataset) ---
HF_REPO_ID   = os.getenv("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
HF_TOKEN     = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))  # peut être None si repo public

PARQUET_NAME = "train_last10w.parquet"
MAX_WEEKS = 10

# ✅ Horizon max pour prédire au-delà du dernier jour observé
FUTURE_DAYS_MAX = 90  # ajuste (30/60/90/365)


# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1500px; }

/* ===== HERO ===== */
.prediction-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 28px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 25px 70px rgba(102, 126, 234, 0.35);
    position: relative;
    overflow: hidden;
}
.prediction-hero::before {
    content: ''; position: absolute; top: -100px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
    border-radius: 50%;
}
.prediction-hero::after {
    content: ''; position: absolute; bottom: -80px; left: -80px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { color: white; font-size: 3.2rem; font-weight: 900; margin: 0 0 1rem 0; letter-spacing: -0.03em; }
.hero-subtitle { color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0; font-weight: 400; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.25);
    backdrop-filter: blur(10px); border-radius: 12px;
    padding: 0.6rem 1.2rem; color: white; font-weight: 700;
    margin-top: 1.5rem; font-size: 0.95rem;
}

/* ===== INFO CARD ===== */
.info-card {
    background: white; border-radius: 20px; padding: 1.8rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}
.info-label { font-size: 0.8rem; color: #999; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.info-value { font-size: 1.6rem; font-weight: 900; color: #667eea; margin-bottom: 0.3rem; }
.info-detail { font-size: 0.85rem; color: #666; }

/* ===== MEGA PRED CARD ===== */
.mega-prediction {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 28px;
    padding: 3rem;
    margin: 2rem 0;
    box-shadow: 0 25px 60px rgba(240, 147, 251, 0.4);
    position: relative;
    overflow: hidden;
}
.prediction-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; position: relative; z-index: 1; }
.prediction-main { color: white; }
.prediction-label { font-size: 1rem; opacity: 0.9; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.1em; }
.prediction-value { font-size: 5.2rem; font-weight: 900; line-height: 1; margin-bottom: 0.8rem; text-shadow: 0 6px 20px rgba(0,0,0,0.2); }
.prediction-unit { font-size: 1.1rem; opacity: 0.95; font-weight: 600; }
.prediction-details { display: grid; gap: 1rem; }
.detail-row {
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.detail-label { font-size: 0.9rem; opacity: 0.9; }
.detail-value { font-size: 1.05rem; font-weight: 800; }

/* ===== METRICS ===== */
.metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin: 2rem 0; }
.metric-box {
    background: white; border-radius: 18px; padding: 1.5rem;
    text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-top: 4px solid;
}
.metric-box:nth-child(1) { border-top-color: #667eea; }
.metric-box:nth-child(2) { border-top-color: #f093fb; }
.metric-box:nth-child(3) { border-top-color: #11998e; }
.metric-icon { font-size: 2.4rem; margin-bottom: 0.6rem; }
.metric-label { font-size: 0.85rem; color: #999; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 2rem; font-weight: 900; color: #111827; }

/* ===== CHART WRAPPER ===== */
.chart-wrapper {
    background: white; border-radius: 20px; padding: 2rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    margin: 1.5rem 0;
}
.chart-title {
    font-size: 1.25rem; font-weight: 900; color: #111827;
    margin-bottom: 1.2rem; padding-bottom: 1rem;
    border-bottom: 3px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD HF ARTIFACTS (model + pipeline + features)
# ============================================================
@st.cache_resource(show_spinner=True)
def load_artifacts_hf():
    return download_artifacts_from_latest(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        hf_token=HF_TOKEN,          # ok même si None (repo public)
        artifacts_dir="artifacts",
        cache_dir=".cache/favorita_artifacts",
    )

try:
    model, pipe, feature_cols, meta = load_artifacts_hf()
    st.caption(
        f"✅ Model HF chargé | run={meta.get('run_id')} | trained_at={meta.get('trained_at', meta.get('updated_at'))}"
    )
except Exception as e:
    st.error("❌ Impossible de charger les artefacts depuis HuggingFace.")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD DATA (HF ONLY) — juste pour listes store/item + min/max
# ============================================================
@st.cache_data(show_spinner=True)
def load_recent_data(weeks: int):
    df_ = load_train_from_hf(weeks=int(weeks), filename=PARQUET_NAME)
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    df_ = df_.dropna(subset=["date"])
    return df_

WEEKS = int(st.session_state.get("weeks_window", MAX_WEEKS))
WEEKS = min(WEEKS, MAX_WEEKS)

df = load_recent_data(WEEKS)

store_list = np.sort(df["store_nbr"].dropna().unique()).tolist()
item_list  = np.sort(df["item_nbr"].dropna().unique()).tolist()

min_d = df["date"].min().date()
max_d = df["date"].max().date()
future_max_d = max_d + timedelta(days=FUTURE_DAYS_MAX)


# ============================================================
# HEADER
# ============================================================
col_hero, col_info = st.columns([0.7, 0.3])

with col_hero:
    st.markdown(f"""
    <div class="prediction-hero">
        <div class="hero-content">
            <div class="hero-title">🔮 Prédictions IA</div>
            <div class="hero-subtitle">Moteur de prévision des ventes (HF artifacts)</div>
            <div class="hero-badge">✨ Source: HuggingFace (100% HF) | Horizon: +{FUTURE_DAYS_MAX} jours</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">Fenêtre de données (pour listes)</div>
        <div class="info-value">{WEEKS} semaines</div>
        <div class="info-detail">📅 Observé: {min_d} → {max_d}</div>
        <div class="info-detail">🚀 Max préd: {future_max_d}</div>
        <div class="info-detail">📦 Parquet: {PARQUET_NAME}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Configuration")

    with st.expander("⚡ Prédiction Instantanée", expanded=True):
        # ✅ Autoriser des dates futures
        date_in = st.date_input("📅 Date", value=max_d, min_value=min_d, max_value=future_max_d)

        store_nbr = st.selectbox("🏪 Store", options=store_list, index=0)

        q = st.text_input("🔍 Rechercher un item", value="", placeholder="ID de l'item...")
        if q.strip():
            item_opts = [x for x in item_list if q.strip() in str(x)][:5000]
        else:
            item_opts = item_list[:5000]

        item_nbr = st.selectbox("📦 Item", options=item_opts, index=0)
        onpromotion = st.checkbox("🏷️ En promotion", value=False)

        st.caption("ℹ️ Astuce: si ton pipeline utilise des lags/rolling, la qualité en futur dépend de la façon dont ces features sont construites.")

    with st.expander("📊 Prédiction sur Période", expanded=False):
        # ✅ Autoriser future range
        date_range = st.date_input(
            "Période",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=future_max_d,
            key="pred_date_range",
        )

        store_sel = st.multiselect("Stores (obligatoire)", options=store_list, default=[])
        item_sel  = st.multiselect("Items (obligatoire)", options=item_opts, default=[])

        promo_mode = st.radio(
            "Promotion (scénario)",
            options=["Toujours NON", "Toujours OUI"],
            index=0,
            horizontal=True,
        )
        promo_value = (promo_mode == "Toujours OUI")

        run_period = st.button("🚀 Lancer Prédiction", width="stretch")


# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["⚡ Instantané", "📈 Période"])


# ============================================================
# TAB 1 — SINGLE PRED
# ============================================================
with tab1:
    new_df = pd.DataFrame({
        "date": [pd.to_datetime(date_in)],
        "store_nbr": [int(store_nbr)],
        "item_nbr": [int(item_nbr)],
        "onpromotion": [bool(onpromotion)],
    })

    X_enriched = pipe.transform(new_df)
    X = (X_enriched
         .reindex(columns=feature_cols, fill_value=0)
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0))

    pred_log = float(model.predict(X)[0])
    pred_sales = float(np.expm1(pred_log))

    st.markdown(f"""
    <div class="mega-prediction">
        <div class="prediction-grid">
            <div class="prediction-main">
                <div class="prediction-label">Prévision estimée</div>
                <div class="prediction-value">{pred_sales:.2f}</div>
                <div class="prediction-unit">unités vendues</div>
            </div>
            <div class="prediction-details">
                <div class="detail-row">
                    <div class="detail-label">📅 Date</div>
                    <div class="detail-value">{pd.to_datetime(date_in).strftime('%d/%m/%Y')}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🏪 Store</div>
                    <div class="detail-value">{store_nbr}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">📦 Item</div>
                    <div class="detail-value">{item_nbr}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🏷️ Promotion</div>
                    <div class="detail-value">{'✅ Oui' if onpromotion else '❌ Non'}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🧮 Log(pred)</div>
                    <div class="detail-value">{pred_log:.4f}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Détails de l'observation"):
        st.dataframe(new_df, width="stretch")


# ============================================================
# TAB 2 — PERIOD PRED (FUTURE READY + 1 CURVE PER ITEM)
# ============================================================
with tab2:
    st.markdown("### 📊 Prédictions sur période (futur inclus)")

    if run_period:
        # --- parse période ---
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_d = pd.to_datetime(date_range[0])
            end_d   = pd.to_datetime(date_range[1])
        else:
            start_d = pd.to_datetime(date_range)
            end_d   = pd.to_datetime(date_range)

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        # ✅ sécurité UX : obliger une sélection (sinon grid énorme)
        if not store_sel:
            st.warning("⚠️ Choisis au moins 1 store.")
            st.stop()
        if not item_sel:
            st.warning("⚠️ Choisis au moins 1 item (pour afficher une courbe par item).")
            st.stop()

        # --- construire un grid (dates x stores x items) ---
        dates = pd.date_range(start_d, end_d, freq="D")

        grid = pd.MultiIndex.from_product(
            [dates, store_sel, item_sel],
            names=["date", "store_nbr", "item_nbr"]
        ).to_frame(index=False)

        grid["onpromotion"] = bool(promo_value)

        # ✅ limiter taille du grid
        nmax = 300_000
        if len(grid) > nmax:
            st.info(f"📊 Grid trop grand ({len(grid):,} lignes). Échantillonnage à {nmax:,}.")
            grid = grid.sample(nmax, random_state=42)

        with st.spinner("⚙️ Prédiction en cours..."):
            Xg_enriched = pipe.transform(grid)
            Xg = (Xg_enriched
                  .reindex(columns=feature_cols, fill_value=0)
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0))

            pred_log_arr = model.predict(Xg)
            pred = np.expm1(pred_log_arr)

        out = grid[["date", "store_nbr", "item_nbr"]].copy()
        out["pred_unit_sales"] = pred.astype("float32")

        total = float(out["pred_unit_sales"].sum())
        avg   = float(out["pred_unit_sales"].mean())
        nrows = int(len(out))

        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Total prédit", f"{total:,.0f}")
        c2.metric("📈 Moyenne / ligne", f"{avg:.2f}")
        c3.metric("🧾 Lignes", f"{nrows:,}")

        # =====================================================
        # ✅ CHART: 1 courbe par item (somme sur stores)
        # =====================================================
        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Évolution temporelle (1 courbe par item)</div>', unsafe_allow_html=True)

        # agrège sur store (si plusieurs stores sélectionnés)
        g = (out
             .groupby(["date", "item_nbr"], as_index=False)["pred_unit_sales"]
             .sum())

        wide = (g.pivot(index="date", columns="item_nbr", values="pred_unit_sales")
                  .sort_index())

        # limiter le nb de courbes affichées (sinon illisible)
        max_lines = 12
        if wide.shape[1] > max_lines:
            st.info(f"🔎 Trop d'items ({wide.shape[1]}). Affichage limité aux {max_lines} premiers.")
            wide = wide.iloc[:, :max_lines]

        st.line_chart(wide, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # Table + export
        # =====================================================
        with st.expander("📄 Table de prédictions (aperçu)"):
            st.dataframe(out.head(200), width="stretch")

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger les prédictions (CSV)",
            data=csv,
            file_name="predictions_favorita.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption("ℹ️ Note: si ton pipeline construit des lags/rolling, la prédiction future peut être moins fiable si ces features ne sont pas calculées à partir d’un historique.")
