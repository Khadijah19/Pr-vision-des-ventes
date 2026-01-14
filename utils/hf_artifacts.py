# utils/hf_artifacts.py
# -*- coding: utf-8 -*-

"""
Téléchargement / publication des artefacts (model, pipeline, features, metadata)
depuis Hugging Face Hub.

✅ Fix Streamlit Cloud (Linux) :
Les .pkl joblib générés sur Windows peuvent contenir des objets pathlib.WindowsPath.
Sur Linux, l'unpickle crash avec :
UnsupportedOperation: cannot instantiate 'WindowsPath' on your system

On applique donc un patch TEMPORAIRE pendant joblib.load() qui remappe
WindowsPath -> PosixPath (et PureWindowsPath -> PurePosixPath) à la fois dans
`pathlib` ET `pathlib._local` (Python 3.12/3.13 utilise ce module interne).
"""

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
def _winpath_compat_patch():
    """
    Patch temporaire pour permettre le chargement de pickles Windows sur Linux.
    On restaure l'état initial à la fin (important pour ne pas "polluer" le process).
    """
    if os.name == "nt":
        # Sur Windows, pas besoin de patch.
        yield
        return

    import pathlib

    # Python 3.12+ : pathlib a un module interne _local où WindowsPath vit réellement.
    try:
        import pathlib._local as plocal  # type: ignore
    except Exception:
        plocal = None

    # Backup des attributs à restaurer
    backup = {}

    def _save(obj, attr):
        if obj is None:
            return
        key = (id(obj), attr)
        if key not in backup and hasattr(obj, attr):
            backup[key] = getattr(obj, attr)

    def _set(obj, attr, value):
        if obj is None:
            return
        if hasattr(obj, attr):
            setattr(obj, attr, value)

    # Sauvegarder
    _save(pathlib, "WindowsPath")
    _save(pathlib, "PureWindowsPath")
    _save(plocal, "WindowsPath")
    _save(plocal, "PureWindowsPath")

    # Appliquer remap
    _set(pathlib, "WindowsPath", pathlib.PosixPath)
    _set(pathlib, "PureWindowsPath", pathlib.PurePosixPath)
    if plocal is not None:
        _set(plocal, "WindowsPath", pathlib.PosixPath)
        _set(plocal, "PureWindowsPath", pathlib.PurePosixPath)

    try:
        yield
    finally:
        # Restaurer
        for (obj_id, attr), val in backup.items():
            # retrouver l'objet à partir de son id est difficile -> on restaure via références connues
            # donc on restaure explicitement sur pathlib + plocal
            pass

        # restauration explicite
        if (id(pathlib), "WindowsPath") in backup:
            pathlib.WindowsPath = backup[(id(pathlib), "WindowsPath")]  # type: ignore
        if (id(pathlib), "PureWindowsPath") in backup:
            pathlib.PureWindowsPath = backup[(id(pathlib), "PureWindowsPath")]  # type: ignore

        if plocal is not None:
            if (id(plocal), "WindowsPath") in backup:
                plocal.WindowsPath = backup[(id(plocal), "WindowsPath")]  # type: ignore
            if (id(plocal), "PureWindowsPath") in backup:
                plocal.PureWindowsPath = backup[(id(plocal), "PureWindowsPath")]  # type: ignore


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

    with _winpath_compat_patch():
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
