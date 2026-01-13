import streamlit as st
from utils.hf_artifacts import read_latest, download_artifacts_from_latest
from utils.training import train_and_publish

st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin — Entraînement & Artefacts (HF-only)")

# Afficher le latest
try:
    latest = read_latest()
    st.success(f"Latest run: {latest['run_id']} (maj: {latest['updated_at']})")
except Exception as e:
    st.warning("Pas de latest.json trouvé pour l’instant.")
    st.caption(str(e))

weeks_window = st.selectbox("Fenêtre d'entraînement (semaines)", [10, 8, 4, 3, 2, 1], index=0)

if st.button("🚀 Retrain + Publish sur HF", use_container_width=True):
    # 1) Récupérer le token proprement (sans crash si secrets absent)
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None

    with st.spinner("Entraînement + publication en cours..."):
        res = train_and_publish(
            weeks_window=int(weeks_window),
            hf_repo_id="khadidia-77/favorita",
            hf_token=hf_token,
        )

    st.success("✅ Terminé ! Nouveau modèle publié.")
    st.json(res["published"])
    st.json(res["train_metrics"])

