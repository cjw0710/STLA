"""Run post-confirmation adapter ablations on MemeTracker train/validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODELS = ("dyhgcn", "disenidp")
ABLATIONS = ("no_environment", "no_prefix", "historical_only")
DEFAULT_SEEDS = (21, 42, 84, 126, 168)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def complete(path: Path, ablation: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol", {})
    return (
        payload.get("status") == "validation_only_strong_backbone_gate"
        and payload.get("adapter_ablation") == ablation
        and protocol.get("test_materialized") is False
        and protocol.get("test_evaluated") is False
        and protocol.get("test_used_for_selection") is False
        and protocol.get("confirmatory_test_reused") is False
    )


def main() -> None:
    args = parse_args()
    baseline_root = ROOT / "artifacts" / "postfreeze_temporal_baselines"
    result_root = ROOT / "artifacts" / "posttest_validation_ablation"
    checkpoint_root = ROOT / "checkpoints" / "posttest_validation_ablation"
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    total = len(args.models) * len(args.ablations) * len(args.seeds)
    index = 0
    for model in args.models:
        for ablation in args.ablations:
            for seed in args.seeds:
                index += 1
                anchor_stem = f"{model}_memetracker_seed{seed}"
                stem = f"{anchor_stem}_{ablation}"
                baseline = baseline_root / f"{anchor_stem}.json"
                result = result_root / f"{stem}.json"
                checkpoint = checkpoint_root / f"{stem}.pt"
                if not baseline.is_file():
                    raise FileNotFoundError(f"missing frozen baseline result: {baseline}")
                if not args.rerun and complete(result, ablation):
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
                        "--ablation",
                        ablation,
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
