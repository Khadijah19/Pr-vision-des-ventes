# scripts/build_models_now.py
# -*- coding: utf-8 -*-
# scripts/build_models_now.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # racine du projet favorita-dashboard-2
sys.path.insert(0, str(ROOT))


import argparse
import json
from pathlib import Path

from utils.training import train_and_save


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, help="Ex: khadidia-77/favorita")
    parser.add_argument("--weeks_window", type=int, default=10)
    parser.add_argument("--parquet_name", type=str, default="train_last10w.parquet")

    parser.add_argument("--artifacts_root", type=str, default="artifacts")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--hf_cache_dir", type=str, default="data/hf_cache")

    parser.add_argument("--total_days", type=int, default=84)
    parser.add_argument("--test_days", type=int, default=14)
    parser.add_argument("--gap_days", type=int, default=3)
    parser.add_argument("--feature_gap_days", type=int, default=3)
    parser.add_argument("--sales_history_days", type=int, default=120)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--force_cache", action="store_true")

    args = parser.parse_args(argv)

    # ✅ fallback si repo_id pas passé
    if not args.repo_id:
        # Mets ici TON repo HF par défaut
        args.repo_id = "khadidia-77/favorita"

    out = train_and_save(
        repo_id=args.repo_id,
        weeks_window=args.weeks_window,
        parquet_name=args.parquet_name,
        artifacts_root=args.artifacts_root,
        models_dir=args.models_dir,
        hf_cache_dir=args.hf_cache_dir,
        total_days=args.total_days,
        test_days=args.test_days,
        gap_days=args.gap_days,
        feature_gap_days=args.feature_gap_days,
        sales_history_days=args.sales_history_days,
        random_state=args.random_state,
        force_cache=args.force_cache,
    )

    latest = out["latest"]
    metrics = out["metrics"]

    print("\n✅ Training terminé")
    print(f"Run id      : {latest['run_id']}")
    print(f"Run dir     : {latest['run_dir']}")
    print(f"Models dir  : {latest['models_dir']}")
    print(f"weeks_window: {latest['weeks_window']}")
    print(f"RMSE_raw    : {metrics.get('RMSE_raw')}")
    print(f"RMSE_log    : {metrics.get('RMSE_log')}")

    Path(args.artifacts_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.artifacts_root) / "build_models_output.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


if __name__ == "__main__":
    main()
