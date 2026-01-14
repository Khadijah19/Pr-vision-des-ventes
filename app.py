# app.py
import streamlit as st
import pandas as pd
import numpy as np

from utils.viz import line_sales_over_time_sum, bar_top_families_sum
from utils.data_loader import load_train_from_hf, load_items_hf, load_stores_hf

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Favorita Forecast Dashboard",
    page_icon="📦",
    layout="wide",
)

PARQUET_NAME = "train_last10w.parquet"
MAX_WEEKS = 10

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1600px; }

/* HERO */
.dashboard-hero {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    border-radius: 30px;
    padding: 4rem 3rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 30px 80px rgba(30, 60, 114, 0.4);
    position: relative;
    overflow: hidden;
}
.dashboard-hero::before{
    content:'';
    position:absolute;
    top:-150px; right:-150px;
    width:600px; height:600px;
    background: radial-gradient(circle, rgba(255,255,255,0.25) 0%, transparent 60%);
    border-radius:50%;
    animation: float 8s ease-in-out infinite;
}
.dashboard-hero::after{
    content:'';
    position:absolute;
    bottom:-100px; left:-100px;
    width:450px; height:450px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius:50%;
    animation: float 10s ease-in-out infinite reverse;
}
@keyframes float { 0%,100%{transform:translate(0,0)} 50%{transform:translate(30px,-30px)} }

.hero-title {
    color:white; font-size:3.5rem; font-weight:900;
    margin:0 0 1rem 0; letter-spacing:-0.04em;
    text-shadow:0 6px 20px rgba(0,0,0,0.2); line-height:1.1;
}
.hero-subtitle { color: rgba(255,255,255,0.95); font-size:1.3rem; margin:0; font-weight:400; line-height:1.6; }

/* KPI */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
    max-width: 700px;
    margin: 0 auto 2rem auto;
}
.kpi-card {
    background: white;
    border-radius: 22px;
    padding: 2rem 1.5rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border-top: 5px solid;
    transition: all 0.25s ease;
}
.kpi-card:nth-child(1) { border-top-color: #1e3c72; }
.kpi-card:nth-child(2) { border-top-color: #2a5298; }
.kpi-card:hover { transform: translateY(-6px); box-shadow: 0 16px 45px rgba(0,0,0,0.14); }
.kpi-icon { font-size: 2.2rem; margin-bottom: 1rem; display: block; }
.kpi-label { font-size: 0.8rem; color: #6b7280; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.6rem; }
.kpi-value { font-size: 2.4rem; font-weight: 900; color: #111827; line-height: 1; }

/* SECTIONS */
.chart-section {
    background: white;
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 8px 35px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.04);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.chart-section::before {
    content: '';
    position: absolute;
    top:0; left:0; right:0; height:5px;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
}
.chart-header {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:2rem; padding-bottom:1.5rem;
    border-bottom:3px solid #f3f4f6;
}
.chart-title {
    font-size:1.6rem; font-weight:900; color:#111827;
    display:flex; align-items:center; gap:0.8rem;
}
.chart-icon {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #fff 100%);
    border-right: 2px solid rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
section[data-testid="stSidebar"] h2 {
    background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip:text;
    font-size: 1.7rem; font-weight: 900;
    margin-bottom: 2rem; padding-bottom: 1rem;
    border-bottom: 3px solid #f3f4f6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="dashboard-hero">
  <div class="hero-title">📦 Favorita Forecast</div>
  <div class="hero-subtitle">
    Tableau de bord analytique avancé pour la prédiction des ventes<br>
    avec visualisations interactives
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONFIG
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    weeks_window = st.selectbox("⏱️ Fenêtre (semaines)", [10, 8, 4, 3, 2, 1], index=0, key="weeks_window")
    st.caption(f"💡 Fenêtre max = {MAX_WEEKS} (parquet HF = {PARQUET_NAME})")
    st.divider()
    st.markdown("## 🎛️ Filtres")
    st.caption("Le graphe journalier s'affiche seulement si tu choisis ≥1 store et ≥1 item.")

# ============================================================
# LOAD DATA (HF ONLY)
# ============================================================
@st.cache_data(show_spinner=True)
def load_all(weeks_window: int):
    train = load_train_from_hf(weeks=int(weeks_window), filename=PARQUET_NAME)
    items = load_items_hf("items.csv")
    stores = load_stores_hf("stores.csv")
    return train, items, stores

train, items, stores = load_all(int(weeks_window))

train["date"] = pd.to_datetime(train["date"], errors="coerce").dt.normalize()
min_d, max_d = train["date"].min(), train["date"].max()

store_list = np.sort(train["store_nbr"].unique()).tolist()
item_list = np.sort(train["item_nbr"].unique()).tolist()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    date_range = st.date_input(
        "📅 Période",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date()
    )

    store_sel = st.multiselect("🏪 Stores", store_list, default=[])

    q_item = st.text_input("🔎 Rechercher un item (id)", value="")
    if q_item.strip():
        item_opts = [x for x in item_list if q_item.strip() in str(x)][:5000]
    else:
        item_opts = item_list[:5000]

    item_sel = st.multiselect("📦 Items", options=item_opts, default=[])

# ============================================================
# APPLY FILTERS
# ============================================================
start_d = pd.to_datetime(date_range[0])
end_d   = pd.to_datetime(date_range[1])

df = train.loc[(train["date"] >= start_d) & (train["date"] <= end_d)].copy()

if store_sel:
    df = df.loc[df["store_nbr"].isin(store_sel)]
if item_sel:
    df = df.loc[df["item_nbr"].isin(item_sel)]

df["unit_sales_pos"] = df["unit_sales"].clip(lower=0)

# ============================================================
# TOP FAMILIES DATASET (indépendant item filter)
# ============================================================
df_base = train.loc[(train["date"] >= start_d) & (train["date"] <= end_d)].copy()
if store_sel:
    df_base = df_base.loc[df_base["store_nbr"].isin(store_sel)]
df_base["unit_sales_pos"] = df_base["unit_sales"].clip(lower=0)

items_min = items[["item_nbr", "family"]].copy()
items_min["item_nbr"] = items_min["item_nbr"].astype("int32", errors="ignore")
items_min["family"] = items_min["family"].fillna("UNKNOWN").astype(str)

# ============================================================
# KPIs
# ============================================================
n_stores = int(df["store_nbr"].nunique())
n_items  = int(df["item_nbr"].nunique())

st.markdown(f"""
<div class="kpi-container">
  <div class="kpi-card">
    <span class="kpi-icon">🏪</span>
    <div class="kpi-label">Stores actifs</div>
    <div class="kpi-value">{n_stores:,}</div>
  </div>
  <div class="kpi-card">
    <span class="kpi-icon">📦</span>
    <div class="kpi-label">Items actifs</div>
    <div class="kpi-value">{n_items:,}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================
left, right = st.columns([2.1, 1])

with left:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown('''
    <div class="chart-header">
      <div class="chart-title">
        <span class="chart-icon">📈</span>
        Ventes journalières (somme unit_sales)
      </div>
    </div>
    ''', unsafe_allow_html=True)

    if (not store_sel) or (not item_sel):
        st.info("ℹ️ Sélectionne au moins **1 store** et **1 item** pour afficher la courbe journalière.")
    else:
        from utils.viz import line_sales_over_time_by_item

        fig1 = line_sales_over_time_by_item(
            df[["date", "item_nbr", "unit_sales_pos"]],
            y_col="unit_sales_pos",
            item_col="item_nbr"
        )

        st.plotly_chart(fig1, width='stretch')

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown('''
    <div class="chart-header">
      <div class="chart-title">
        <span class="chart-icon">🧺</span>
        Top familles (somme unit_sales)
      </div>
    </div>
    ''', unsafe_allow_html=True)

    df_fam = df_base[["item_nbr", "unit_sales_pos"]].merge(items_min, on="item_nbr", how="left", copy=False)
    fig2 = bar_top_families_sum(df_fam, y_col="unit_sales_pos", top=10)
    st.plotly_chart(fig2, width='stretch')

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 Aperçu des données filtrées"):
        st.dataframe(df.head(50), width='stretch')

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
  <p style="font-size: 0.9rem; margin: 0;">
    <strong>Favorita Forecast Dashboard</strong> · Propulsé par Streamlit
  </p>
  <p style="font-size: 0.8rem; margin: 0.5rem 0 0 0; opacity: 0.8;">
    © 2026 · Tous droits réservés · Made with ❤️
  </p>
</div>
""", unsafe_allow_html=True)
