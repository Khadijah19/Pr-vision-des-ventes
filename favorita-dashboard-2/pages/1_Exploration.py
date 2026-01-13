# pages/1_📊_Exploration.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.data_loader import load_train_recent, load_items, load_stores

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Exploration", page_icon="📊", layout="wide")

# ============================================================
# DESIGN PREMIUM (UNE SEULE FOIS ✅)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1600px; }

.exploration-hero{
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  border-radius: 24px; padding: 3rem 2.5rem; margin-bottom: 2rem;
  box-shadow: 0 20px 60px rgba(17,153,142,0.25);
  position: relative; overflow: hidden;
}
.exploration-hero::before{
  content:''; position:absolute; bottom:-120px; left:-120px;
  width:420px; height:420px;
  background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, transparent 70%);
  border-radius:50%;
}
.exploration-hero h1{ color:#fff; font-size:2.8rem; font-weight:900; margin:0 0 .5rem 0; letter-spacing:-0.02em; }
.exploration-hero p{ color:rgba(255,255,255,0.95); font-size:1.05rem; margin:0; font-weight:500; }

.kpi-grid{
  display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.2rem; margin: 1.8rem 0 2rem 0;
}
.kpi-card{
  background:#fff; border-radius:20px; padding:1.6rem;
  box-shadow:0 6px 26px rgba(0,0,0,0.08);
  border-left:5px solid #667eea;
  position:relative; overflow:hidden; transition: all .25s ease;
}
.kpi-card::before{
  content:''; position:absolute; top:-20px; right:-20px; width:120px; height:120px;
  background: radial-gradient(circle, rgba(17,153,142,0.10) 0%, transparent 70%);
  border-radius:50%;
}
.kpi-card:hover{ transform: translateY(-6px); box-shadow:0 14px 40px rgba(0,0,0,0.14); }
.kpi-label{
  font-size:.82rem; color:#6b7280; font-weight:800;
  text-transform:uppercase; letter-spacing:.08em; margin-bottom:.7rem;
}
.kpi-value{ font-size:2.2rem; font-weight:900; color:#111827; line-height:1; }

.chart-container{
  background:#fff; border-radius:20px; padding:2rem;
  box-shadow:0 6px 26px rgba(0,0,0,0.06);
  border:1px solid rgba(0,0,0,0.04);
  margin: 1.2rem 0;
}
.chart-title{
  font-size:1.25rem; font-weight:900; color:#111827;
  margin-bottom:1.2rem; padding-bottom:1rem;
  border-bottom:2px solid #f3f4f6;
}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
  border-right:2px solid rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] .block-container{ padding-top: 1.5rem; }

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  font-weight:900; margin-bottom:1.2rem; padding-bottom:.8rem;
  border-bottom:3px solid #f3f4f6; letter-spacing:-0.02em;
}

section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stDateInput > div > div{
  border-radius:14px; border:2px solid #e5e7eb; background:#fff;
  box-shadow:0 2px 10px rgba(0,0,0,0.03); transition:all .25s ease;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
section[data-testid="stSidebar"] .stDateInput > div > div:focus-within{
  border-color:#667eea; box-shadow:0 0 0 4px rgba(102,126,234,0.12); transform: translateY(-1px);
}

div.stButton > button{
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  color:white; border:none; border-radius:12px;
  padding:0.7rem 1.5rem; font-weight:800;
  box-shadow:0 4px 16px rgba(17,153,142,0.25);
  transition: all .25s ease;
}
div.stButton > button:hover{ transform: translateY(-2px); box-shadow:0 10px 24px rgba(17,153,142,0.35); }

.stTabs [data-baseweb="tab-list"]{
  gap: .5rem; background:#fff; border-radius:16px; padding:.5rem;
  box-shadow:0 2px 12px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"]{ border-radius:12px; padding:.75rem 1.5rem; font-weight:800; color:#6b7280; }
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color:#fff !important; box-shadow:0 6px 18px rgba(102,126,234,0.35);
}

hr{
  border:none; height:2px;
  background: linear-gradient(90deg, transparent, #11998e, transparent);
  margin: 2.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="exploration-hero">
  <h1>📊 Exploration des Données</h1>
  <p>KPIs, tendances, promo uplift, heatmaps et Pareto pour des insights “retail”</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================
DATA_DIR = st.session_state.get("data_dir", "data/favorita_data")
weeks_window = st.session_state.get("weeks_window", 10)

df = load_train_recent(DATA_DIR, weeks=weeks_window)
items = load_items(DATA_DIR)
stores = load_stores(DATA_DIR)

# Sanity
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
df["unit_sales_clean"] = df["unit_sales"].clip(lower=0)
df["onpromotion"] = df["onpromotion"].fillna(False).astype(bool)
df["onpromo"] = df["onpromotion"].astype("int8")

# merge metadata (ok ici, fenêtre déjà réduite)
df = df.merge(items[["item_nbr", "family", "perishable"]], on="item_nbr", how="left")
df = df.merge(stores[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left")

df["family"] = df["family"].fillna("UNKNOWN").astype(str)
df["state"] = df["state"].fillna("UNKNOWN").astype(str)
df["type"] = df["type"].fillna("UNKNOWN").astype(str)
df["city"] = df["city"].fillna("UNKNOWN").astype(str)

# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ Filtres")

    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("📅 Période", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    states = ["ALL"] + sorted(df["state"].unique().tolist())
    state = st.selectbox("🗺️ State", states, index=0)

    df_state = df[df["state"] == state] if state != "ALL" else df

    store_list = ["ALL"] + sorted(df_state["store_nbr"].dropna().unique().tolist())
    store = st.selectbox("🏪 Store", store_list, index=0)

    families = ["ALL"] + sorted(df_state["family"].unique().tolist())
    family = st.selectbox("🏷️ Family", families, index=0)

    promo_mode = st.selectbox("🎁 Promo", ["ALL", "Promo", "Sans promo"], index=0)

# ============================================================
# APPLY FILTERS
# ============================================================
d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
f = df[(df["date"] >= d1) & (df["date"] <= d2)].copy()

if state != "ALL":
    f = f[f["state"] == state]
if store != "ALL":
    # store est int dans la liste -> ok
    f = f[f["store_nbr"] == store]
if family != "ALL":
    f = f[f["family"] == family]
if promo_mode == "Promo":
    f = f[f["onpromo"] == 1]
elif promo_mode == "Sans promo":
    f = f[f["onpromo"] == 0]

if len(f) == 0:
    st.warning("Aucune donnée après filtres. Essaie d’élargir la période ou enlever un filtre.")
    st.stop()

# ============================================================
# KPIs (retail-friendly ✅)
# ============================================================
total_sales = float(f["unit_sales_clean"].sum())
promo_sales = float(f.loc[f["onpromo"] == 1, "unit_sales_clean"].sum())
promo_share = (promo_sales / total_sales) if total_sales > 0 else 0.0

n_stores = int(f["store_nbr"].nunique())
n_items = int(f["item_nbr"].nunique())
n_fam = int(f["family"].nunique())

kpis = [
    ("Ventes totales (sum)", f"{total_sales:,.0f}"),
    ("Part ventes en promo", f"{promo_share*100:.1f}%"),
    ("Stores actifs", f"{n_stores:,}"),
    ("Items actifs", f"{n_items:,}"),
    ("Familles actives", f"{n_fam:,}"),
]
import html
import math

palette = ["#667eea", "#11998e", "#f093fb", "#fa709a", "#764ba2"]

def _unwrap(x):
    """Réduit pd.Series/pd.DataFrame/numpy scalars -> python scalar."""
    try:
        import pandas as pd
        import numpy as np
        if isinstance(x, pd.Series):
            x = x.iloc[0] if len(x) else ""
        elif isinstance(x, pd.DataFrame):
            x = x.iloc[0, 0] if (x.shape[0] and x.shape[1]) else ""
        elif isinstance(x, np.generic):
            x = x.item()
    except Exception:
        pass
    return x

def _format_value(v):
    """
    - int -> '12 345'
    - float:
        * si 0<=v<=1 => pourcentage '20.8%'
        * sinon => '12 345.67' (ou '12 346' si proche int)
    """
    v = _unwrap(v)

    # bool / None
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Oui" if v else "Non"

    # essayer numeric
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return "—"

        # % si valeur entre 0 et 1 (ex: promo_rate)
        if 0 <= fv <= 1:
            return f"{fv*100:.1f}%"

        # si proche d'un entier, afficher comme int
        if abs(fv - round(fv)) < 1e-9:
            s = f"{int(round(fv)):,}".replace(",", " ")
            return s

        # sinon float arrondi
        s = f"{fv:,.2f}".replace(",", " ")
        return s
    except Exception:
        # fallback texte
        return str(v)

def _safe(s):
    return html.escape(str(s))

kpi_html = '<div class="kpi-grid">'
for i, (k, v) in enumerate(kpis[:5]):
    border = palette[i % len(palette)]
    k_s = _safe(k)
    v_s = _safe(_format_value(v))

    kpi_html += f"""
    <div class="kpi-card" style="border-left-color:{border};">
        <div class="kpi-label">{k_s}</div>
        <div class="kpi-value">{v_s}</div>
    </div>
    """
kpi_html += "</div>"

st.markdown(kpi_html, unsafe_allow_html=True)


# ============================================================
# TABS (PLUS “BUSINESS” ✅)
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Tendances", "🏷️ Promotions (Uplift)", "🔥 Heatmaps", "🏆 Top & Pareto"]
)

# ---------- TAB 1: Tendances ----------
with tab1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📈 Ventes journalières + moyenne mobile (7 jours)</div>', unsafe_allow_html=True)

    g = f.groupby("date", as_index=False)["unit_sales_clean"].sum().sort_values("date")
    g["ma7"] = g["unit_sales_clean"].rolling(7, min_periods=1).mean()

    fig = px.line(g, x="date", y=["unit_sales_clean", "ma7"], title="")
    fig.update_layout(template="plotly_white", height=420, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 Distribution des ventes (échantillon)</div>', unsafe_allow_html=True)

    sample = f["unit_sales_clean"].sample(min(len(f), 200_000), random_state=42)
    fig2 = px.histogram(sample, nbins=120, title="")
    fig2.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB 2: Promotions ----------
with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📈 Ventes journalières : Promo vs Sans promo</div>', unsafe_allow_html=True)

    g = f.groupby(["date", "onpromo"], as_index=False)["unit_sales_clean"].sum()
    g["onpromo"] = g["onpromo"].map({0: "Sans promo", 1: "Promo"})

    fig = px.line(g, x="date", y="unit_sales_clean", color="onpromo", title="")
    fig.update_layout(template="plotly_white", height=420, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🚀 Promo uplift par famille (Top 20)</div>', unsafe_allow_html=True)

    tmp = f.groupby(["family", "onpromo"], as_index=False)["unit_sales_clean"].mean()
    p = tmp.pivot(index="family", columns="onpromo", values="unit_sales_clean").fillna(0)

    # colonnes possibles (0/1)
    if 0 not in p.columns:
        p[0] = 0.0
    if 1 not in p.columns:
        p[1] = 0.0

    p = p.rename(columns={0: "Sans promo", 1: "Promo"})
    p["uplift_%"] = np.where(p["Sans promo"] > 0, (p["Promo"] / p["Sans promo"] - 1) * 100, 0.0)
    p = p.sort_values("uplift_%", ascending=False).head(20)

    fig = px.bar(p.reset_index(), x="uplift_%", y="family", orientation="h", title="")
    fig.update_layout(template="plotly_white", height=560, yaxis_title="", xaxis_title="Uplift promo (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB 3: Heatmaps ----------
with tab3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔥 Heatmap : Jour de semaine × Mois (sum ventes)</div>', unsafe_allow_html=True)

    tmp = f.copy()
    tmp["dow"] = tmp["date"].dt.day_name()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)

    h = tmp.groupby(["dow", "month"], as_index=False)["unit_sales_clean"].sum()

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    h["dow"] = pd.Categorical(h["dow"], categories=dow_order, ordered=True)

    pivot = h.pivot(index="dow", columns="month", values="unit_sales_clean").fillna(0)

    fig = px.imshow(pivot, aspect="auto", title="")
    fig.update_layout(template="plotly_white", height=520, xaxis_title="Mois", yaxis_title="Jour")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📍 Répartition des ventes par type de store</div>', unsafe_allow_html=True)

    gt = f.groupby("type", as_index=False)["unit_sales_clean"].sum().sort_values("unit_sales_clean", ascending=False)
    fig = px.bar(gt, x="type", y="unit_sales_clean", title="")
    fig.update_layout(template="plotly_white", height=420, xaxis_title="Type", yaxis_title="Ventes (sum)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB 4: Top & Pareto ----------
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🏪 Top Stores (stacked : Promo vs Sans promo)</div>', unsafe_allow_html=True)

        top_store_ids = (
            f.groupby("store_nbr")["unit_sales_clean"].sum().sort_values(ascending=False).head(15).index.tolist()
        )

        gs = f.groupby(["store_nbr", "onpromo"], as_index=False)["unit_sales_clean"].sum()
        gs = gs[gs["store_nbr"].isin(top_store_ids)]
        gs["onpromo"] = gs["onpromo"].map({0: "Sans promo", 1: "Promo"})

        fig = px.bar(gs, x="store_nbr", y="unit_sales_clean", color="onpromo", title="")
        fig.update_layout(template="plotly_white", height=460, xaxis_title="Store", yaxis_title="Ventes (sum)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🧺 Pareto des familles (Top 25)</div>', unsafe_allow_html=True)

        gf = f.groupby("family", as_index=False)["unit_sales_clean"].sum().sort_values("unit_sales_clean", ascending=False)
        gf["cum_pct"] = gf["unit_sales_clean"].cumsum() / gf["unit_sales_clean"].sum() * 100
        gf25 = gf.head(25)

        fig_bar = px.bar(gf25, x="family", y="unit_sales_clean", title="")
        fig_bar.update_layout(template="plotly_white", height=420, xaxis_tickangle=-35, xaxis_title="", yaxis_title="Ventes (sum)")
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_line = px.line(gf25, x="family", y="cum_pct", title="")
        fig_line.update_layout(template="plotly_white", height=300, xaxis_tickangle=-35, xaxis_title="", yaxis_title="Cumul (%)")
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔍 Aperçu des données filtrées</div>', unsafe_allow_html=True)
    st.dataframe(f.head(60), use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("💡 Idées next level : impact jours fériés (holidays_events), volatilité (CV), saisonnalité (Fourier), et ‘promo elasticity’ par famille/store.")
