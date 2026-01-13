# utils/hf_artifacts.py
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from contextlib import contextmanager
import importlib

import joblib
from huggingface_hub import hf_hub_download, HfApi

HF_DATASET_REPO = "khadidia-77/favorita"
HF_REPO_TYPE = "dataset"

ARTIFACTS_ROOT = "artifacts"
LATEST_PATH = f"{ARTIFACTS_ROOT}/latest.json"


def _run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@contextmanager
def _winpath_compat_patch():
    """
    Patch ultra-robuste pour charger un joblib/pickle créé sur Windows
    dans un environnement Linux (Streamlit Cloud).

    Python 3.13 peut instancier via pathlib._local.WindowsPath.__new__
    qui lève UnsupportedOperation sur Linux.
    -> On patch :
       - les symboles WindowsPath/PureWindowsPath (pathlib + pathlib._local)
       - ET le __new__ de WindowsPath pour ne plus lever.
    """
    if os.name == "nt":
        yield
        return

    import pathlib

    # Import sûr de pathlib._local (py3.13)
    try:
        plocal = importlib.import_module("pathlib._local")
    except Exception:
        plocal = None

    # Sauvegarde états
    old = {}

    def _save_attr(mod, name):
        if mod is None:
            return
        key = (mod.__name__, name)
        old[key] = getattr(mod, name, None)

    def _set_attr(mod, name, value):
        if mod is None:
            return
        try:
            setattr(mod, name, value)
        except Exception:
            pass

    # sauvegarde avant patch
    _save_attr(pathlib, "WindowsPath")
    _save_attr(pathlib, "PureWindowsPath")
    if plocal is not None:
        _save_attr(plocal, "WindowsPath")
        _save_attr(plocal, "PureWindowsPath")

    # aussi sauvegarder __new__ si possible
    try:
        if plocal is not None and getattr(plocal, "WindowsPath", None) is not None:
            _save_attr(plocal.WindowsPath, "__new__")
        if getattr(pathlib, "WindowsPath", None) is not None:
            _save_attr(pathlib.WindowsPath, "__new__")
    except Exception:
        pass

    try:
        # 1) Remap les classes
        _set_attr(pathlib, "WindowsPath", pathlib.PosixPath)
        _set_attr(pathlib, "PureWindowsPath", pathlib.PurePosixPath)
        if plocal is not None:
            _set_attr(plocal, "WindowsPath", pathlib.PosixPath)
            _set_attr(plocal, "PureWindowsPath", pathlib.PurePosixPath)

        # 2) Patch __new__ (c'est ça qui te bloque dans le traceback)
        # Si pickle résout encore WindowsPath, il n'aura plus le __new__ "qui raise".
        try:
            if plocal is not None and hasattr(plocal, "WindowsPath"):
                plocal.WindowsPath.__new__ = pathlib.PosixPath.__new__  # type: ignore
        except Exception:
            pass

        try:
            if hasattr(pathlib, "WindowsPath"):
                pathlib.WindowsPath.__new__ = pathlib.PosixPath.__new__  # type: ignore
        except Exception:
            pass

        yield

    finally:
        # restore propre
        for (modname, attr), val in old.items():
            try:
                mod = importlib.import_module(modname)
                setattr(mod, attr, val)
            except Exception:
                pass


def read_latest(
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = HF_REPO_TYPE,
    hf_token: Optional[str] = None,
    artifacts_dir: str = ARTIFACTS_ROOT,
) -> Dict:
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
    latest = read_latest(
        repo_id=repo_id,
        repo_type=repo_type,
        hf_token=hf_token,
        artifacts_dir=artifacts_dir,
    )
    run_dir = latest["run_dir"]

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

    # ✅ patch pendant l'unpickle/joblib.load
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
