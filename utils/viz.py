# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 15:58:53 2026

@author: HP
"""

# utils/viz.py
import pandas as pd
import plotly.express as px


def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
        title_font=dict(size=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


# -----------------------------
# Helpers internes (sans changer les imports)
# -----------------------------
def _fallback_sales_col(df, y_col):
    if y_col in df.columns:
        return y_col
    for alt in ["unit_sales_pos", "unit_sales_clean", "unit_sales"]:
        if alt in df.columns:
            return alt
    raise ValueError(f"Colonne ventes introuvable: '{y_col}' (et pas de fallback).")


def _ensure_date(df, date_col):
    if date_col not in df.columns:
        raise ValueError(f"Colonne date introuvable: '{date_col}'")
    d = pd.to_datetime(df[date_col], errors="coerce")
    return d


# -----------------------------
# 1) Line: ventes par jour
# -----------------------------
def line_sales_over_time(df, date_col="date", y_col="unit_sales", mode="count", title=None):
    """
    mode="count": ventes = nb d'observations par jour
    mode="units": ventes = somme(y_col) par jour
    """
    d = _ensure_date(df, date_col)
    tmp = df.loc[d.notna(), [date_col]].copy()
    tmp[date_col] = d.loc[d.notna()].values

    if mode == "count":
        agg = tmp.groupby(date_col).size().reset_index(name="n_obs")
        fig = px.line(agg, x=date_col, y="n_obs", title=title or "Ventes par jour (nb obs)")
        return apply_plotly_theme(fig)

    # mode units
    y_col = _fallback_sales_col(df, y_col)
    tmp = df.loc[d.notna(), [date_col, y_col]].copy()
    tmp[date_col] = d.loc[d.notna()].values
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").fillna(0).clip(lower=0)

    agg = tmp.groupby(date_col, as_index=False)[y_col].sum()
    fig = px.line(agg, x=date_col, y=y_col, title=title or "Unités vendues par jour")
    return apply_plotly_theme(fig)


# -----------------------------
# 2) Bar top families
# -----------------------------
def bar_top_families(df_merge_items, y_col="unit_sales", top=10, mode="count", title=None):
    """
    mode="count": top familles par nb obs
    mode="units": top familles par somme(y_col)
    """
    if "family" not in df_merge_items.columns:
        raise ValueError("Colonne 'family' manquante (merge items nécessaire).")

    if mode == "count":
        agg = (df_merge_items.groupby("family", as_index=False)
               .size()
               .rename(columns={"size": "n_obs"})
               .sort_values("n_obs", ascending=False)
               .head(int(top)))
        fig = px.bar(agg, x="family", y="n_obs", title=title or f"Top {top} familles (nb obs)")
        return apply_plotly_theme(fig)

    y_col = _fallback_sales_col(df_merge_items, y_col)
    tmp = df_merge_items[["family", y_col]].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").fillna(0).clip(lower=0)

    agg = (tmp.groupby("family", as_index=False)[y_col]
           .sum()
           .sort_values(y_col, ascending=False)
           .head(int(top)))
    fig = px.bar(agg, x="family", y=y_col, title=title or f"Top {top} familles (unités vendues)")
    return apply_plotly_theme(fig)


# -----------------------------
# 3) KPI (déjà ok)
# -----------------------------
def kpi_card_df(df, date_col="date", sales_col="unit_sales_pos", mode="count"):
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=[date_col])

    total_obs = int(len(df2))
    if total_obs == 0:
        out = {"KPI": ["Période"], "Valeur": ["—"]}
        return pd.DataFrame(out)

    if mode == "count":
        per_day = df2.groupby(date_col).size()
        avg_per_day = float(per_day.mean()) if len(per_day) else 0.0

        out = {
            "KPI": ["Période", "Nb stores", "Nb items", "Ventes totales (nb obs)", "Ventes moy./jour (nb obs)"],
            "Valeur": [
                f"{df2[date_col].min().date()} → {df2[date_col].max().date()}",
                int(df2["store_nbr"].nunique()) if "store_nbr" in df2.columns else 0,
                int(df2["item_nbr"].nunique()) if "item_nbr" in df2.columns else 0,
                int(total_obs),
                float(avg_per_day),
            ],
        }
        return pd.DataFrame(out)

    # mode units
    if sales_col not in df2.columns:
        # fallback si besoin
        sales_col = _fallback_sales_col(df2, sales_col)

    s = pd.to_numeric(df2[sales_col], errors="coerce").fillna(0).clip(lower=0)
    total_units = float(s.sum())
    avg_units_day = float(df2.assign(_sales=s).groupby(date_col)["_sales"].sum().mean())

    out = {
        "KPI": ["Période", "Nb stores", "Nb items", "Unités vendues (somme)", "Unités vendues moy./jour"],
        "Valeur": [
            f"{df2[date_col].min().date()} → {df2[date_col].max().date()}",
            int(df2["store_nbr"].nunique()) if "store_nbr" in df2.columns else 0,
            int(df2["item_nbr"].nunique()) if "item_nbr" in df2.columns else 0,
            float(total_units),
            float(avg_units_day),
        ],
    }
    return pd.DataFrame(out)


# -----------------------------
# 4) Alias line_sales_by_day (compat)
# -----------------------------
def line_sales_by_day(df, y_col="unit_sales_clean", y=None, date_col="date", title="Ventes totales par jour", mode="count"):
    if y is not None:
        y_col = y
    # title passe dans line_sales_over_time
    return line_sales_over_time(df, date_col=date_col, y_col=y_col, mode=mode, title=title)


# -----------------------------
# 5) Bar promo (même logique count/units)
# -----------------------------
def bar_sales_promo(
    df,
    promo_col="onpromotion",
    y_col="unit_sales",
    y=None,
    title="Ventes: Promo vs Pas promo",
    mode="count",
):
    """
    mode="count": nb obs selon promo
    mode="units": somme(y_col) selon promo
    """
    if y is not None:
        y_col = y

    # fallback ventes (seulement si mode units)
    if mode != "count":
        y_col = _fallback_sales_col(df, y_col)

    # fallback promo
    if promo_col not in df.columns:
        if "onpromo" in df.columns:
            promo_col = "onpromo"
        else:
            raise ValueError(f"Colonne promo introuvable: '{promo_col}'")

    promo = df[promo_col]
    # normaliser promo en 0/1
    if promo.dtype == "bool":
        promo01 = promo.fillna(False).astype(bool).astype("int8")
    else:
        promo01 = pd.to_numeric(promo, errors="coerce").fillna(0).astype("int8").clip(0, 1)

    labels = promo01.map({0: "Sans promo", 1: "Avec promo"})

    if mode == "count":
        agg = labels.value_counts().reset_index()
        agg.columns = ["promo", "n_obs"]
        fig = px.bar(agg, x="promo", y="n_obs", title=title)
        return apply_plotly_theme(fig)

    sales = pd.to_numeric(df[y_col], errors="coerce").fillna(0).clip(lower=0)
    agg = sales.groupby(labels).sum().reset_index()
    agg.columns = ["promo", y_col]

    fig = px.bar(agg, x="promo", y=y_col, title=title)
    return apply_plotly_theme(fig)


# -----------------------------
# 6) Heatmap (même logique count/units)
# -----------------------------
def heatmap_dow_week(
    df,
    date_col="date",
    y_col="unit_sales",
    y=None,
    title="Heatmap: ventes par semaine x jour",
    mode="count",
):
    """
    mode="count": nb obs par (semaine ISO, jour)
    mode="units": somme(y_col) par (semaine ISO, jour)
    """
    if y is not None:
        y_col = y

    d = _ensure_date(df, date_col)
    tmp = df.loc[d.notna()].copy()
    tmp[date_col] = d.loc[d.notna()].values

    iso = tmp[date_col].dt.isocalendar()
    tmp["iso_year"] = iso["year"].astype(int)
    tmp["iso_week"] = iso["week"].astype(int)
    tmp["dow"] = tmp[date_col].dt.weekday

    if mode == "count":
        agg = (tmp.groupby(["iso_year", "iso_week", "dow"], as_index=False)
               .size()
               .rename(columns={"size": "n_obs"}))
        value_col = "n_obs"
    else:
        y_col = _fallback_sales_col(tmp, y_col)
        tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").fillna(0).clip(lower=0)
        agg = tmp.groupby(["iso_year", "iso_week", "dow"], as_index=False)[y_col].sum()
        value_col = y_col

    pivot = agg.pivot_table(index=["iso_year", "iso_week"], columns="dow", values=value_col, fill_value=0)
    pivot = pivot.reindex(columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0)
    pivot.columns = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    pivot = pivot.reset_index()
    pivot["week_label"] = pivot["iso_year"].astype(str) + "-W" + pivot["iso_week"].astype(str).str.zfill(2)

    fig = px.imshow(
        pivot[["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]],
        x=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
        y=pivot["week_label"],
        aspect="auto",
        title=title,
    )
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=55, b=10))
    return fig


# -----------------------------
# 7) Top table (même logique count/units)
# -----------------------------
def top_n_table(
    df,
    group_col,
    y_col="unit_sales",
    y=None,
    top=10,
    n=None,
    ascending=False,
    mode="count",
):
    if y is not None:
        y_col = y
    if n is not None:
        top = n

    if group_col not in df.columns:
        raise ValueError(f"Colonne group_col introuvable: '{group_col}'")

    if mode == "count":
        out = (
            df.groupby(group_col, as_index=False)
              .size()
              .rename(columns={"size": "n_obs"})
              .sort_values("n_obs", ascending=ascending)
              .head(int(top))
              .reset_index(drop=True)
        )
        return out

    # mode units
    y_col = _fallback_sales_col(df, y_col)
    tmp = df[[group_col, y_col]].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").fillna(0).clip(lower=0)

    out = (
        tmp.groupby(group_col, as_index=False)[y_col]
          .sum()
          .sort_values(y_col, ascending=ascending)
          .head(int(top))
          .reset_index(drop=True)
    )
    return out



def line_sales_over_time_sum(df, y_col="unit_sales_pos"):
    # somme des ventes par jour
    g = (df.groupby("date", as_index=False)[y_col]
           .sum()
           .sort_values("date"))
    fig = px.line(g, x="date", y=y_col, title="Ventes par jour (somme)")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig

def line_sales_over_time_by_group(df, group_col, y_col="unit_sales_pos", top_n=8):
    # top groupes par somme totale
    totals = df.groupby(group_col)[y_col].sum().sort_values(ascending=False)
    top_groups = totals.head(top_n).index

    f = df[df[group_col].isin(top_groups)].copy()

    g = (f.groupby(["date", group_col], as_index=False)[y_col]
           .sum()
           .sort_values("date"))

    fig = px.line(g, x="date", y=y_col, color=group_col,
                  title=f"Ventes par jour (somme) — Top {top_n} {group_col}")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text=str(group_col))
    return fig

def bar_top_families_sum(df_fam, y_col="unit_sales_pos", top=10):
    g = (df_fam.groupby("family", as_index=False)[y_col]
            .sum()
            .sort_values(y_col, ascending=False)
            .head(top))

    fig = px.bar(g, x=y_col, y="family", orientation="h",
                 title=f"Top {top} familles (somme unit_sales)")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis=dict(categoryorder="total ascending"))
    return fig

import plotly.express as px
import pandas as pd

def line_sales_over_time_by_item(df: pd.DataFrame, date_col="date", y_col="unit_sales_pos", item_col="item_nbr"):
    """
    Courbe journalière avec une ligne par item.
    df doit contenir: date_col, item_col, y_col
    """
    g = (df.groupby([date_col, item_col], as_index=False)[y_col]
           .sum()
           .sort_values(date_col))

    fig = px.line(
        g,
        x=date_col,
        y=y_col,
        color=item_col,  # ✅ 1 courbe par item
        markers=False,
        title=""
    )
    fig.update_layout(template="plotly_white", height=430, legend_title_text="Item")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Ventes (sum)")
    return fig
