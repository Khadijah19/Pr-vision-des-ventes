# pages/2_Prédictions.py
# -*- coding: utf-8 -*-




import os
import json
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_train_from_hf
from utils.hf_artifacts import download_artifacts_from_latest, HF_DATASET_REPO




HF_REPO_ID = st.secrets.get("HF_REPO_ID", "khadidia-77/favorita")
HF_REPO_TYPE = st.secrets.get("HF_REPO_TYPE", "dataset")

# token optionnel (repo public => None)
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)
except Exception:
    HF_TOKEN = None

@st.cache_resource
def load_artifacts_hf():
    return download_artifacts_from_latest(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        hf_token=HF_TOKEN,
        artifacts_dir="artifacts",
    )

model, pipe, feature_cols, meta = load_artifacts_hf()
st.caption(f"✅ Model HF chargé | run={meta.get('run_id')} | trained_at={meta.get('trained_at')}")


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Prédictions - Favorita",
    page_icon="🔮",
    layout="wide",
)

PARQUET_NAME = "train_last10w.parquet"
MAX_WEEKS = 10

# Repo HF (dataset) : priorité session_state -> défaut constant
HF_REPO_ID = st.session_state.get("hf_repo_id", HF_DATASET_REPO)


# ============================================================
# SECRETS SAFE (ne crash pas si pas de secrets.toml)
# ============================================================
def get_hf_token() -> str | None:
    # 1) secrets.toml (si existe)
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok)
    except Exception:
        pass

    # 2) variable d'environnement
    return os.environ.get("HF_TOKEN") or None


HF_TOKEN = get_hf_token()


# ============================================================
# CSS (ONE TIME, TOP OF FILE)
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

/* ===== SIDEBAR CLEAN PREMIUM ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9ff 0%, #fff 100%);
    border-right: 1px solid rgba(0,0,0,0.08);
}
section[data-testid="stSidebar"] h2 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.4rem;
    font-weight: 900;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD HF ARTIFACTS (100% HF)
# ============================================================
@st.cache_resource(show_spinner=True)
def load_hf_artifacts(repo_id: str, hf_token: str | None):
    # télécharge latest.json puis run artifacts
    model, pipe, feature_cols, metadata = download_artifacts_from_latest(
        repo_id=repo_id,
        hf_token=hf_token,  # nécessite la petite modif dans utils/hf_artifacts.py (je te la donne en bas)
    )
    return model, pipe, feature_cols, metadata


try:
    model, pipe, feature_cols, meta = load_hf_artifacts(HF_REPO_ID, HF_TOKEN)
except Exception as e:
    st.error("❌ Impossible de charger les artefacts depuis HuggingFace.")
    st.code(str(e))
    st.info(
        "✅ Vérifie : (1) le repo_id, (2) si le repo est privé → token HF requis, "
        "(3) que artifacts/latest.json existe dans le dataset."
    )
    st.stop()


# ============================================================
# LOAD DATA (HF ONLY)
# ============================================================
@st.cache_data(show_spinner=True)
def load_recent_data(repo_id: str, hf_token: str | None, weeks: int):
    df_ = load_train_from_hf(
        repo_id=repo_id,
        hf_token=hf_token,
        weeks=int(weeks),
        filename=PARQUET_NAME,
    )
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    df_ = df_.dropna(subset=["date"])
    return df_


WEEKS = int(st.session_state.get("weeks_window", MAX_WEEKS))
WEEKS = min(WEEKS, MAX_WEEKS)

df = load_recent_data(HF_REPO_ID, HF_TOKEN, WEEKS)

store_list = np.sort(df["store_nbr"].dropna().unique()).tolist()
item_list  = np.sort(df["item_nbr"].dropna().unique()).tolist()

min_d = df["date"].min().date()
max_d = df["date"].max().date()


# ============================================================
# HEADER
# ============================================================
col_hero, col_info = st.columns([0.7, 0.3])

with col_hero:
    st.markdown(f"""
    <div class="prediction-hero">
        <div class="hero-content">
            <div class="hero-title">🔮 Prédictions IA</div>
            <div class="hero-subtitle">Moteur de prévision des ventes (HF latest artifacts)</div>
            <div class="hero-badge">✨ Repo: {HF_REPO_ID}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">Fenêtre de données</div>
        <div class="info-value">{WEEKS} semaines</div>
        <div class="info-detail">📅 {min_d} → {max_d}</div>
        <div class="info-detail">📦 Parquet: {PARQUET_NAME}</div>
        <div class="info-detail">🧠 Run: {meta.get("run_id", meta.get("updated_at", "latest"))}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Configuration")

    with st.expander("⚙️ HuggingFace", expanded=False):
        st.write("Source : artifacts/latest.json → dernier run")
        st.caption("Si repo privé, ajoute HF_TOKEN (secrets.toml ou variable d'env).")

        if st.button("🔄 Recharger (vider cache)", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with st.expander("⚡ Prédiction Instantanée", expanded=True):
        date_in = st.date_input("📅 Date", value=max_d, min_value=min_d, max_value=max_d)
        store_nbr = st.selectbox("🏪 Store", options=store_list, index=0)

        q = st.text_input("🔍 Rechercher un item", value="", placeholder="ID de l'item...")
        if q.strip():
            item_opts = [x for x in item_list if q.strip() in str(x)][:5000]
        else:
            item_opts = item_list[:5000]

        item_nbr = st.selectbox("📦 Item", options=item_opts, index=0)
        onpromotion = st.checkbox("🏷️ En promotion", value=False)

    with st.expander("📊 Prédiction sur Période", expanded=False):
        date_range = st.date_input(
            "Période",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="pred_date_range",
        )
        store_sel = st.multiselect("Stores", options=store_list, default=[])
        item_sel  = st.multiselect("Items", options=item_opts, default=[])
        run_period = st.button("🚀 Lancer Prédiction", use_container_width=True)


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

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Ventes prévues</div>
            <div class="metric-value">{pred_sales:.2f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-icon">🧪</div>
            <div class="metric-label">Log prédiction</div>
            <div class="metric-value">{pred_log:.4f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-icon">🏷️</div>
            <div class="metric-label">Promo</div>
            <div class="metric-value">{'OUI' if onpromotion else 'NON'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Détails de l'observation"):
        st.dataframe(new_df, use_container_width=True)

    st.caption(f"✅ Model loaded from HF | trained_at={meta.get('trained_at')} | run={meta.get('run_id', 'latest')}")


# ============================================================
# TAB 2 — PERIOD PRED
# ============================================================
with tab2:
    st.markdown("### 📊 Prédictions sur période avec filtres")

    if run_period:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_d = pd.to_datetime(date_range[0])
            end_d   = pd.to_datetime(date_range[1])
        else:
            start_d = pd.to_datetime(date_range)
            end_d   = pd.to_datetime(date_range)

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        f = df.loc[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()

        if store_sel:
            f = f.loc[f["store_nbr"].isin(store_sel)]
        if item_sel:
            f = f.loc[f["item_nbr"].isin(item_sel)]

        if len(f) == 0:
            st.warning("⚠️ Aucune ligne après filtres.")
            st.stop()

        nmax = 300_000
        if len(f) > nmax:
            f = f.sample(nmax, random_state=42)
            st.info(f"📊 Dataset échantillonné : {nmax:,} lignes")

        with st.spinner("⚙️ Prédiction en cours..."):
            Xf_enriched = pipe.transform(f)
            Xf = (Xf_enriched
                  .reindex(columns=feature_cols, fill_value=0)
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0))

            pred_log_arr = model.predict(Xf)
            pred = np.expm1(pred_log_arr)

        out = f[["date", "store_nbr", "item_nbr"]].copy()
        out["pred_unit_sales"] = pred.astype("float32")

        total = float(out["pred_unit_sales"].sum())
        avg   = float(out["pred_unit_sales"].mean())
        nrows = int(len(out))

        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-icon">📦</div>
                <div class="metric-label">Total prédit</div>
                <div class="metric-value">{total:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">📈</div>
                <div class="metric-label">Moyenne / ligne</div>
                <div class="metric-value">{avg:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">🧾</div>
                <div class="metric-label">Lignes</div>
                <div class="metric-value">{nrows:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Évolution temporelle (total prédit)</div>', unsafe_allow_html=True)
        g1 = out.groupby("date", as_index=False)["pred_unit_sales"].sum()
        st.line_chart(g1.set_index("date"))
        st.markdown('</div>', unsafe_allow_html=True)

        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">🏪 Top Stores</div>', unsafe_allow_html=True)
            g2 = (out.groupby("store_nbr", as_index=False)["pred_unit_sales"]
                    .sum().sort_values("pred_unit_sales", ascending=False).head(15))
            st.bar_chart(g2.set_index("store_nbr"))
            st.markdown('</div>', unsafe_allow_html=True)

        with cB:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">📦 Top Items</div>', unsafe_allow_html=True)
            g3 = (out.groupby("item_nbr", as_index=False)["pred_unit_sales"]
                    .sum().sort_values("pred_unit_sales", ascending=False).head(15))
            st.bar_chart(g3.set_index("item_nbr"))
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 Table de prédictions (aperçu)"):
            st.dataframe(out.head(200), use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger les prédictions (CSV)",
            data=csv,
            file_name="predictions_favorita.csv",
            mime="text/csv",
            use_container_width=True,
        )
