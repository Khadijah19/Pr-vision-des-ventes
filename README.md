# Favorita Forecast Dashboard

Tableau de bord interactif pour explorer les ventes récentes du dataset **Corporación Favorita Grocery Sales Forecasting** et générer des **prévisions de ventes** à partir d’artefacts de modèle publiés sur **Hugging Face** (modèle + pipeline de features + liste de features).

---

## Accès rapide

- **Dashboard (Streamlit Cloud)**  
  https://dashboard-favorita-pbnsy5hoyafjvoklvssvtg.streamlit.app/

- **Dépôt principal (code source officiel)**  
  Ce dépôt est un **fork**. Le dépôt principal du dashboard est ici :  
  https://github.com/Khadijah19/Dashboard_-Favorita

---

## Fonctionnalités

### Page d’accueil (Analytics)
- Exploration des ventes sur une fenêtre récente (10 semaines)
- Filtres par période, store(s) et item(s)
- Visualisations :
  - courbes temporelles
  - top familles (via jointure avec `items.csv`)
- Aperçu des données filtrées

### Page Prédictions
- **Prévision ponctuelle** : date + store + item + promotion
- **Prévision sur période** : prédictions sur une plage de dates, avec filtres
- Export des résultats en **CSV**

### Page Admin
- Suivi de l’état des artefacts publiés (latest run)
- Évaluation simple des performances sur une fenêtre récente (validation sur les derniers jours)

---

## Architecture du projet

```text
.
├── app.py
├── pages/
│   ├── 0_Admin.py
│   └── 2_Predictions.py
├── utils/
│   ├── data_loader.py
│   ├── hf_artifacts.py
│   └── viz.py
├── artifacts/                 # optionnel en local (sinon téléchargé depuis HF)
├── requirements.txt
└── README.md
```

---

## Données et modèle

### Données
Les données récentes utilisées par l’application sont chargées depuis **Hugging Face** au format Parquet.

- Exemple : `train_last10w.parquet` (fenêtre de 10 semaines)
- Fichiers de référence :
  - `items.csv`
  - `stores.csv`

### Modèle

## Entraînement automatisé (GitHub Actions)

L’entraînement et la génération des artefacts (modèle + pipeline + liste de features) sont automatisés via **GitHub Actions**.
À chaque exécution du workflow, les artefacts sont publiés sur **Hugging Face** puis consommés par l’application Streamlit (mode "HF-only").

L’application récupère automatiquement les derniers artefacts disponibles :
- `model` (régression)
- `pipe` (Feature Pipeline)
- `feature_cols` (liste des features attendues)
- `meta` (informations du run)

---

## Configuration (variables d’environnement / secrets)

L’application utilise ces paramètres :

- `HF_REPO_ID` : identifiant du repo HF (ex: `khadidia-77/favorita`)
- `HF_REPO_TYPE` : type de repo (souvent `dataset`)
- `HF_TOKEN` : token Hugging Face (optionnel si repo public)
- `PARQUET_NAME` : nom du parquet (ex: `train_last10w.parquet`)

### Sur Streamlit Cloud
Ajoute `HF_TOKEN` dans **Settings → Secrets** :

```toml
HF_TOKEN = "xxxxx"
HF_REPO_ID = "khadidia-77/favorita"
HF_REPO_TYPE = "dataset"
```

---

## Installation en local

### 1) Cloner le dépôt
```bash
git clone https://github.com/Khadijah19/Dashboard_-Favorita
cd Dashboard_-Favorita
```

### 2) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3) Lancer l’application
```bash
streamlit run app.py
```

---

## Notes importantes

- Les pages utilisent une fenêtre récente (10 semaines) afin de réduire la charge mémoire et améliorer la stabilité.
- Les prédictions sont calculées en `log1p`, puis reconverties en unités via `expm1`.
- Si vous utilisez un repo HF privé, `HF_TOKEN` est requis.

---

## Crédits

- Dataset : *Corporación Favorita Grocery Sales Forecasting* (Kaggle)
- Déploiement : Streamlit Cloud
- Artefacts : Hugging Face

---

## Licence

Ce projet est proposé à des fins éducatives et de démonstration.  
Vérifiez les licences associées au dataset et aux dépendances utilisées.
