"""Run the seed-21 strong-backbone temporal-adapter validation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = ("christian", "android", "douban", "twitter", "memetracker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-model", default="dyhgcn", choices=("dyhgcn", "mshgat", "disenidp"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seed", type=int, default=None, help="backward-compatible single seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol", {})
    return (
        payload.get("status") == "validation_only_strong_backbone_gate"
        and protocol.get("test_materialized") is False
        and protocol.get("test_evaluated") is False
        and protocol.get("test_used_for_selection") is False
    )


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 21]
    baseline_root = ROOT / "artifacts" / "postfreeze_temporal_baselines"
    result_root = ROOT / "artifacts" / "postfreeze_strong_adapter"
    checkpoint_root = ROOT / "checkpoints" / "postfreeze_strong_adapter"
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    total = len(args.datasets) * len(seeds)
    index = 0
    for dataset in args.datasets:
        for seed in seeds:
            index += 1
            stem = f"{args.anchor_model}_{dataset}_seed{seed}"
            baseline = baseline_root / f"{stem}.json"
            result = result_root / f"{stem}.json"
            checkpoint = checkpoint_root / f"{stem}.pt"
            if not baseline.is_file():
                raise FileNotFoundError(f"missing frozen baseline result: {baseline}")
            if not args.rerun and complete(result):
                print(f"[{index}/{total}] reuse {stem}", flush=True)
                continue
            print(f"[{index}/{total}] run {stem}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "www2027.train_strong_logit_adapter",
                    "--baseline-result",
                    str(baseline),
                    "--checkpoint",
                    str(checkpoint),
                    "--result-json",
                    str(result),
                ],
                cwd=ROOT.parent,
                check=True,
            )


if __name__ == "__main__":
    main()
