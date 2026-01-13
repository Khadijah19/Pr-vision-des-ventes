# utils/hf_artifacts.py
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from contextlib import contextmanager

import joblib
from huggingface_hub import hf_hub_download, HfApi

HF_DATASET_REPO = "khadidia-77/favorita"   # dataset HF
HF_REPO_TYPE = "dataset"

ARTIFACTS_ROOT = "artifacts"
LATEST_PATH = f"{ARTIFACTS_ROOT}/latest.json"


def _run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@contextmanager
def _patch_windows_path_for_linux():
    """
    Fix robuste pour charger des pickles/joblib créés sur Windows
    dans un environnement Linux (Streamlit Cloud).

    On patch:
    - pathlib.WindowsPath / PureWindowsPath
    - pathlib._local.WindowsPath / PureWindowsPath (important en py3.13)
    """
    if os.name == "nt":
        yield
        return

    import pathlib

    old_pathlib_windows = getattr(pathlib, "WindowsPath", None)
    old_pathlib_pure_windows = getattr(pathlib, "PureWindowsPath", None)

    # Python 3.13: certaines classes peuvent venir de pathlib._local
    try:
        import pathlib._local as plocal  # type: ignore
    except Exception:
        plocal = None

    old_local_windows = getattr(plocal, "WindowsPath", None) if plocal else None
    old_local_pure_windows = getattr(plocal, "PureWindowsPath", None) if plocal else None

    try:
        # Patch "public"
        pathlib.WindowsPath = pathlib.PosixPath          # type: ignore
        pathlib.PureWindowsPath = pathlib.PurePosixPath  # type: ignore

        # Patch "internal" (py3.13)
        if plocal is not None:
            plocal.WindowsPath = pathlib.PosixPath          # type: ignore
            plocal.PureWindowsPath = pathlib.PurePosixPath  # type: ignore

        yield

    finally:
        # restore
        if old_pathlib_windows is not None:
            pathlib.WindowsPath = old_pathlib_windows  # type: ignore
        if old_pathlib_pure_windows is not None:
            pathlib.PureWindowsPath = old_pathlib_pure_windows  # type: ignore

        if plocal is not None:
            if old_local_windows is not None:
                plocal.WindowsPath = old_local_windows  # type: ignore
            if old_local_pure_windows is not None:
                plocal.PureWindowsPath = old_local_pure_windows  # type: ignore


def read_latest(
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = HF_REPO_TYPE,
    hf_token: Optional[str] = None,
    artifacts_dir: str = ARTIFACTS_ROOT,
) -> Dict:
    """
    Lit artifacts/latest.json depuis HF.
    """
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=f"{artifacts_dir}/latest.json",
        token=hf_token,
    )
    with open(local, "r", encoding="utf-8") as f:
        return json.load(f)


def download_artifacts_from_latest(
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = HF_REPO_TYPE,
    hf_token: Optional[str] = None,
    artifacts_dir: str = ARTIFACTS_ROOT,
    cache_dir: str = ".cache/favorita_artifacts",
) -> Tuple[object, object, List[str], Dict]:
    """
    Télécharge les 4 artefacts du run pointé par latest.json, puis charge en mémoire.
    Retourne: model, pipeline, feature_cols, metadata
    """
    latest = read_latest(
        repo_id=repo_id,
        repo_type=repo_type,
        hf_token=hf_token,
        artifacts_dir=artifacts_dir,
    )

    run_dir = latest["run_dir"]  # ex: artifacts/runs/2026-01-13_07-12-05

    def dl(name: str) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"{run_dir}/{name}",
            cache_dir=cache_dir,
            token=hf_token,
        )

    model_path = dl("best_model.pkl")
    pipe_path  = dl("feature_pipeline.pkl")
    feat_path  = dl("features.json")
    meta_path  = dl("metadata.json")

    # ✅ patch uniquement pendant l'unpickle
    with _patch_windows_path_for_linux():
        model = joblib.load(model_path)
        pipe  = joblib.load(pipe_path)

    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, pipe, feature_cols, metadata


def publish_run_to_hf(
    local_models_dir: str = "models",
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = HF_REPO_TYPE,
    hf_token: Optional[str] = None,
    artifacts_dir: str = ARTIFACTS_ROOT,
) -> Dict:
    """
    Publie les fichiers générés dans models/ vers HF dans un nouveau run,
    puis met à jour latest.json.
    """
    local_models_dir = Path(local_models_dir)
    required = ["best_model.pkl", "feature_pipeline.pkl", "features.json", "metadata.json"]

    for f in required:
        if not (local_models_dir / f).exists():
            raise FileNotFoundError(f"Fichier manquant: {local_models_dir / f}")

    api = HfApi(token=hf_token)

    run_id = _run_id()
    run_dir = f"{artifacts_dir}/runs/{run_id}"

    for f in required:
        api.upload_file(
            path_or_fileobj=str(local_models_dir / f),
            path_in_repo=f"{run_dir}/{f}",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add artifacts run {run_id}",
        )

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
        path_in_repo=f"{artifacts_dir}/latest.json",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Update latest -> {run_id}",
    )

    try:
        tmp.unlink()
    except Exception:
        pass

    return latest_payload
