"""Run the ten frozen MemeTracker strong-adapter one-shot evaluations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ("dyhgcn", "disenidp")
SEEDS = (21, 42, 84, 126, 168)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=ROOT / "artifacts" / "memetracker_pretest_selection_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = 0
    for model in MODELS:
        for seed in SEEDS:
            index += 1
            stem = f"{model}_memetracker_seed{seed}"
            print(f"[{index}/10] evaluate {stem}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "www2027.evaluate_memetracker_strong_test",
                    "--baseline-result",
                    str(ROOT / "artifacts" / "postfreeze_temporal_baselines" / f"{stem}.json"),
                    "--adapter-result",
                    str(ROOT / "artifacts" / "postfreeze_strong_adapter" / f"{stem}.json"),
                    "--selection-manifest",
                    str(args.selection_manifest),
                    "--result-json",
                    str(ROOT / "artifacts" / "memetracker_one_shot" / f"{stem}.json"),
                ],
                cwd=ROOT.parent,
                check=True,
            )


if __name__ == "__main__":
    main()
