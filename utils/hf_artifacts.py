# utils/hf_artifacts.py
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

# utils/hf_artifacts.py
import os
import pathlib

# ✅ Compat: artefacts picklés sous Windows -> chargés sous Linux (Streamlit Cloud)
if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath
    pathlib.PureWindowsPath = pathlib.PurePosixPath

    # Python 3.13 peut instancier via le module interne pathlib._local
    try:
        import pathlib._local as _local
        _local.WindowsPath = pathlib.PosixPath
        _local.PureWindowsPath = pathlib.PurePosixPath
    except Exception:
        pass


import joblib
from huggingface_hub import hf_hub_download, HfApi

HF_DATASET_REPO = "khadidia-77/favorita"   # ton dataset HF
HF_REPO_TYPE = "dataset"

ARTIFACTS_ROOT = "artifacts"              # dans le dataset
LATEST_PATH = f"{ARTIFACTS_ROOT}/latest.json"


def _run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def read_latest(repo_id: str = HF_DATASET_REPO) -> Dict:
    """
    Lit artifacts/latest.json depuis HF.
    """
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        filename=LATEST_PATH,
    )
    with open(local, "r", encoding="utf-8") as f:
        return json.load(f)


# utils/hf_artifacts.py
# -*- coding: utf-8 -*-


DEFAULT_REPO_ID = "khadidia-77/favorita"
DEFAULT_REPO_TYPE = "dataset"


def _download_json(repo_id: str, filename: str, repo_type: str, token: str | None):
    fp = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        token=token,
    )
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def download_artifacts_from_latest(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    hf_token: str | None = None,
    artifacts_dir: str = "artifacts",
):
    """
    Télécharge depuis HF (dataset) :
      artifacts/latest.json
      + les artefacts du run pointé (runs/<run_id>/...)

    Retourne: model, pipe, feature_cols, meta
    """
    # 1) lire latest.json
    latest_path = f"{artifacts_dir}/latest.json"
    latest = _download_json(repo_id, latest_path, repo_type, hf_token)

    run_id = latest.get("run_id")
    if not run_id:
        raise FileNotFoundError(f"latest.json ne contient pas run_id: {latest}")

    run_prefix = f"{artifacts_dir}/runs/{run_id}"

    # 2) télécharger les fichiers du run
    model_fp = hf_hub_download(
        repo_id=repo_id, repo_type=repo_type,
        filename=f"{run_prefix}/best_model.pkl",
        token=hf_token,
    )
    pipe_fp = hf_hub_download(
        repo_id=repo_id, repo_type=repo_type,
        filename=f"{run_prefix}/feature_pipeline.pkl",
        token=hf_token,
    )
    feat_fp = hf_hub_download(
        repo_id=repo_id, repo_type=repo_type,
        filename=f"{run_prefix}/features.json",
        token=hf_token,
    )
    meta_fp = hf_hub_download(
        repo_id=repo_id, repo_type=repo_type,
        filename=f"{run_prefix}/metadata.json",
        token=hf_token,
    )

    model = joblib.load(model_fp)
    pipe = joblib.load(pipe_fp)
    with open(feat_fp, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(meta_fp, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # enrich meta
    meta["run_id"] = run_id
    meta["repo_id"] = repo_id
    meta["repo_type"] = repo_type

    return model, pipe, feature_cols, meta


def publish_run_to_hf(
    local_models_dir: str = "models",
    repo_id: str = HF_DATASET_REPO,
    hf_token: str | None = None,
) -> Dict:
    """
    Publie les fichiers générés dans models/ vers HF dans un nouveau run,
    puis met à jour latest.json (atomicité logique).
    """
    local_models_dir = Path(local_models_dir)
    required = ["best_model.pkl", "feature_pipeline.pkl", "features.json", "metadata.json"]

    for f in required:
        if not (local_models_dir / f).exists():
            raise FileNotFoundError(f"Fichier manquant: {local_models_dir / f}")

    api = HfApi(token=hf_token)
    run_id = _run_id()
    run_dir = f"{ARTIFACTS_ROOT}/runs/{run_id}"

    # 1) upload des 4 artefacts dans runs/<run_id>/
    for f in required:
        api.upload_file(
            path_or_fileobj=str(local_models_dir / f),
            path_in_repo=f"{run_dir}/{f}",
            repo_id=repo_id,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Add artifacts run {run_id}",
        )

    # 2) upload latest.json (petit) en dernier = “switch” vers ce run
    latest_payload = {
        "run_id": run_id,
        "run_dir": run_dir,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    tmp = local_models_dir / "_latest.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=LATEST_PATH,
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        commit_message=f"Update latest -> {run_id}",
    )

    try:
        tmp.unlink()
    except Exception:
        pass

    return latest_payload
