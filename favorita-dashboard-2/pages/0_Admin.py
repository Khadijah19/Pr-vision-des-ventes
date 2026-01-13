# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 07:27:12 2026

@author: HP
"""

# pages/0_Admin.py
import streamlit as st
from pathlib import Path

from utils.data_loader import load_metadata, load_metrics
from utils.training import train_and_save

st.set_page_config(page_title="Admin - Favorita", page_icon="🛠️", layout="wide")

MODELS_DIR = Path("models")

st.title("🛠️ Admin — Ré-entraîner le modèle (HF)")

st.info("✅ Mode **HF-only** : l’app lit les données depuis HuggingFace. L’entraînement ne se fait que si tu cliques sur le bouton.")

# --- Show current metadata/metrics
meta = load_metadata(MODELS_DIR)
metrics = load_metrics(MODELS_DIR)

c1, c2, c3 = st.columns(3)
c1.metric("Source", meta.get("source", "—"))
c2.metric("Weeks", str(meta.get("weeks", "—")))
c3.metric("Max date (train)", meta.get("max_date", "—"))

with st.expander("📦 Artifacts actuels"):
    st.write("metadata.json")
    st.json(meta if meta else {})
    st.write("metrics.json")
    st.json(metrics if metrics else {})

st.divider()

# --- Training controls
st.subheader("🚀 Lancer un nouvel entraînement")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    weeks = st.number_input("Fenêtre (semaines)", min_value=2, max_value=10, value=10, step=1)
with colB:
    val_days = st.number_input("Validation (derniers jours)", min_value=1, max_value=21, value=7, step=1)
with colC:
    parquet_name = st.text_input("Parquet HF", value="train_last10w.parquet")

danger = st.checkbox("⚠️ Je comprends que ça écrase l’ancien modèle", value=False)

if st.button("🧠 Réentraîner maintenant", type="primary", use_container_width=True, disabled=not danger):
    with st.spinner("Entraînement en cours..."):
        try:
            new_metrics = train_and_save(
                weeks_window=int(weeks),
                parquet_name=parquet_name.strip(),
                models_dir=MODELS_DIR,
                val_days=int(val_days),
            )
            st.success("✅ Entraînement terminé. Nouveaux artifacts sauvegardés dans /models.")
            st.json(new_metrics)
            st.balloons()
        except Exception as e:
            st.error(f"❌ Erreur pendant l'entraînement : {e}")
