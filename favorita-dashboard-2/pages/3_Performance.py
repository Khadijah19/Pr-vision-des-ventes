#pages/3_Performance
import streamlit as st
from utils.model_loader import load_metrics

st.set_page_config(page_title="Performance", layout="wide")
st.title("📊 Performance du modèle")

m = load_metrics()
if not m:
    st.warning("Aucune métrique trouvée. Va dans Admin → retrain / build_models_now.")
    st.stop()

c1, c2, c3 = st.columns(3)
# si metadata.json => RMSE_log / MAE_log / R2_log
c1.metric("RMSE_log", f"{m.get('RMSE_log', 0):.4f}")
c2.metric("MAE_log", f"{m.get('MAE_log', 0):.4f}")
c3.metric("R²_log", f"{m.get('R2_log', 0):.4f}")

st.json(m)


# À ajouter dans CHAQUE page (app.py, 0_Admin.py, 1_Exploration.py, 2_Prédictions.py)
# Remplace ou ajoute ce CSS dans la section st.markdown() existante

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ===== SIDEBAR ULTRA PREMIUM ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    border-right: 2px solid rgba(0,0,0,0.06);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ===== SIDEBAR TITLES ===== */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 3px solid #f3f4f6;
    letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] h2 {
    font-size: 1.4rem;
}

/* ===== SIDEBAR SECTIONS ===== */
section[data-testid="stSidebar"] > div > div > div > div {
    padding: 0.5rem 0;
}

/* ===== INPUTS PREMIUM ===== */
section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stDateInput > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div {
    border-radius: 14px;
    border: 2px solid #e5e7eb;
    background: white;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

section[data-testid="stSidebar"] .stTextInput > div > div:focus-within,
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
section[data-testid="stSidebar"] .stDateInput > div > div:focus-within,
section[data-testid="stSidebar"] .stMultiSelect > div > div:focus-within,
section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within {
    border-color: #667eea;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.12);
    transform: translateY(-1px);
}

/* ===== LABELS PREMIUM ===== */
section[data-testid="stSidebar"] label {
    font-weight: 600 !important;
    color: #374151 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* ===== CHECKBOX PREMIUM ===== */
section[data-testid="stSidebar"] .stCheckbox {
    background: white;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] .stCheckbox:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(255,255,255,1) 100%);
}

/* ===== RADIO BUTTONS PREMIUM ===== */
section[data-testid="stSidebar"] .stRadio {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    border: 2px solid #f3f4f6;
}

section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 700 !important;
    color: #1f2937 !important;
    margin-bottom: 1rem !important;
}

section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.5rem;
}

section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
    margin: 0.3rem 0;
}

section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
    transform: translateX(4px);
}

/* ===== SLIDER PREMIUM ===== */
section[data-testid="stSidebar"] .stSlider {
    padding: 1rem;
    background: white;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

section[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: linear-gradient(90deg, #1e3c72 0%, #7e22ce 100%) !important;
}

section[data-testid="stSidebar"] .stSlider > div > div > div > div > div {
    background: white !important;
    border: 3px solid #667eea !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

/* ===== MULTISELECT PREMIUM ===== */
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.7rem !important;
    font-weight: 600 !important;
}

/* ===== EXPANDER PREMIUM ===== */
section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    font-weight: 700;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
    color: #1f2937;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
    transform: translateX(4px);
}

section[data-testid="stSidebar"] .streamlit-expanderContent {
    background: white;
    border: 2px solid #f3f4f6;
    border-top: none;
    border-radius: 0 0 14px 14px;
    padding: 1rem;
}

/* ===== CAPTION / INFO TEXT ===== */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #6b7280 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0;
    display: block;
}

/* ===== DIVIDER PREMIUM ===== */
section[data-testid="stSidebar"] hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, transparent);
    margin: 1.5rem 0;
    border-radius: 2px;
}

/* ===== FILE UPLOADER PREMIUM ===== */
section[data-testid="stSidebar"] .stFileUploader {
    background: white;
    border-radius: 14px;
    border: 2px dashed #e5e7eb;
    padding: 1rem;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] .stFileUploader:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, white 100%);
}

/* ===== BUTTON IN SIDEBAR ===== */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    font-size: 0.95rem;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
    margin-top: 0.5rem;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* ===== ANIMATIONS ===== */
@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

section[data-testid="stSidebar"] > div > div > div > div {
    animation: slideInLeft 0.4s ease-out backwards;
}

section[data-testid="stSidebar"] > div > div > div > div:nth-child(1) { animation-delay: 0.1s; }
section[data-testid="stSidebar"] > div > div > div > div:nth-child(2) { animation-delay: 0.15s; }
section[data-testid="stSidebar"] > div > div > div > div:nth-child(3) { animation-delay: 0.2s; }
section[data-testid="stSidebar"] > div > div > div > div:nth-child(4) { animation-delay: 0.25s; }
section[data-testid="stSidebar"] > div > div > div > div:nth-child(5) { animation-delay: 0.3s; }

/* ===== SCROLLBAR PREMIUM ===== */
section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 8px;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
    background: #f3f4f6;
    border-radius: 10px;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
}
</style>
""", unsafe_allow_html=True)