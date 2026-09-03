"""Run corrected DeDiff anchors and internal dynamic adapters reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = ("christian", "android")
SEEDS = (21, 42, 84, 126, 168)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temporal-rank", type=int, default=8)
    parser.add_argument("--rerun-anchor", action="store_true")
    parser.add_argument("--rerun-dynamic", action="store_true")
    return parser.parse_args()


def complete(path: Path, status: str, dataset: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol", {})
    return (
        payload.get("status") == status
        and payload.get("dataset") == dataset
        and int(payload.get("seed", -1)) == seed
        and protocol.get("test_materialized") is False
        and protocol.get("test_evaluated") is False
        and protocol.get("test_used_for_selection") is False
    )


def main() -> None:
    args = parse_args()
    anchor_results = ROOT / "artifacts" / "postfreeze_temporal_dediff"
    anchor_checkpoints = ROOT / "checkpoints" / "postfreeze_temporal_dediff"
    dynamic_results = ROOT / "artifacts" / "postfreeze_dynamic_dediff"
    dynamic_checkpoints = ROOT / "checkpoints" / "postfreeze_dynamic_dediff"
    for path in (anchor_results, anchor_checkpoints, dynamic_results, dynamic_checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    total = len(args.datasets) * len(args.seeds)
    current = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            current += 1
            anchor_stem = f"dediff_{dataset}_seed{seed}"
            anchor_result = anchor_results / f"{anchor_stem}.json"
            anchor_checkpoint = anchor_checkpoints / f"{anchor_stem}.pt"
            if args.rerun_anchor or not complete(
                anchor_result,
                "validation_only_corrected_dediff",
                dataset,
                seed,
            ):
                print(f"[{current}/{total}] train anchor {anchor_stem}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "www2027.train_temporal_dediff",
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--minimum-epochs",
                        str(args.minimum_epochs),
                        "--patience",
                        str(args.patience),
                        "--steps-per-epoch",
                        str(args.steps_per_epoch),
                        "--batch-size",
                        str(args.batch_size),
                        "--checkpoint",
                        str(anchor_checkpoint),
                        "--result-json",
                        str(anchor_result),
                    ],
                    cwd=ROOT.parent,
                    check=True,
                )
            else:
                print(f"[{current}/{total}] reuse anchor {anchor_stem}", flush=True)

            dynamic_stem = f"dediff_{dataset}_rank{args.temporal_rank}_seed{seed}"
            dynamic_result = dynamic_results / f"{dynamic_stem}.json"
            dynamic_checkpoint = dynamic_checkpoints / f"{dynamic_stem}.pt"
            if args.rerun_dynamic or not complete(
                dynamic_result,
                "validation_only_dynamic_internal_dediff",
                dataset,
                seed,
            ):
                print(f"[{current}/{total}] train dynamic {dynamic_stem}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "www2027.train_dynamic_dediff_adapter",
                        "--anchor-result",
                        str(anchor_result),
                        "--temporal-rank",
                        str(args.temporal_rank),
                        "--epochs",
                        str(args.epochs),
                        "--minimum-epochs",
                        str(args.minimum_epochs),
                        "--patience",
                        str(args.patience),
                        "--steps-per-epoch",
                        str(args.steps_per_epoch),
                        "--batch-size",
                        str(args.batch_size),
                        "--checkpoint",
                        str(dynamic_checkpoint),
                        "--result-json",
                        str(dynamic_result),
                    ],
                    cwd=ROOT.parent,
                    check=True,
                )
            else:
                print(f"[{current}/{total}] reuse dynamic {dynamic_stem}", flush=True)


if __name__ == "__main__":
    main()

