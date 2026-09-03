"""Run temporal BuzzBloom baselines without touching the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODELS = ("DyHGCN", "MSHGAT", "DisenIDP")
DATASETS = ("christian", "android", "douban", "twitter", "memetracker")
SEEDS = (21, 42, 84, 126, 168)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "artifacts" / "postfreeze_temporal_baselines",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "postfreeze_temporal_baselines",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def is_complete(path: Path, model: str, dataset: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol", {})
    return (
        payload.get("model_name") == model
        and payload.get("dataset") == dataset
        and payload.get("seed") == seed
        and protocol.get("test_materialized") is False
        and protocol.get("test_evaluated") is False
        and protocol.get("test_used_for_selection") is False
    )


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total = len(args.models) * len(args.datasets) * len(args.seeds)
    completed = 0
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                stem = f"{model.lower()}_{dataset}_seed{seed}"
                result_path = args.result_dir / f"{stem}.json"
                checkpoint_path = args.checkpoint_dir / f"{stem}.pt"
                completed += 1
                if not args.rerun and is_complete(result_path, model, dataset, seed):
                    print(f"[{completed}/{total}] reuse {stem}", flush=True)
                    continue
                print(f"[{completed}/{total}] run {stem}", flush=True)
                command = [
                    sys.executable,
                    "-m",
                    "www2027.baselines.buzzbloom_temporal",
                    "--model-name",
                    model,
                    "--dataset",
                    dataset,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--minimum-epochs",
                    str(args.minimum_epochs),
                    "--early-stopping-patience",
                    str(args.patience),
                    "--steps-per-epoch",
                    str(args.steps_per_epoch),
                    "--batch-size",
                    str(args.batch_size),
                    "--eval-batch-size",
                    str(args.batch_size),
                    "--d-model",
                    str(args.dimension),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--result-json",
                    str(result_path),
                    "--quiet",
                ]
                subprocess.run(command, cwd=ROOT.parent, check=True)


if __name__ == "__main__":
    main()
