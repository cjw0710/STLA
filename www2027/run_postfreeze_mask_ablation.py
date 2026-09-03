"""Train static-mask and no-mask anchors under the final validation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


MASK_MODES = ("static", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["christian", "android", "douban", "twitter"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[21, 42, 84, 126, 168]
    )
    parser.add_argument("--mask-modes", nargs="+", choices=MASK_MODES, default=list(MASK_MODES))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints"
        / "postfreeze_mask_ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "postfreeze_mask_ablation_training",
    )
    return parser.parse_args()


def annotate_protocol(path: Path) -> None:
    item = json.loads(path.read_text(encoding="utf-8"))
    protocol = item.setdefault("protocol", {})
    protocol.update(
        {
            "split": "chronological_70_10_20_timestamp_ties_preserved",
            "batch_size": 64,
            "max_prefix_length": 50,
            "train_environments": 4,
            "valid_environments": 2,
            "test_environments": 3,
            "sample_hop": 2,
            "max_eval_batches": 0,
            "test_materialized": False,
            "selection_changes_permitted": False,
            "postfreeze_descriptive_ablation": True,
        }
    )
    path.write_text(json.dumps(item, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for dataset in args.datasets:
        for mask_mode in args.mask_modes:
            for seed in args.seeds:
                result_path = args.output_dir / f"{dataset}_{mask_mode}_s{seed}.json"
                checkpoint_path = (
                    args.checkpoint_dir / f"{dataset}_{mask_mode}_anchor_s{seed}.pt"
                )
                if not result_path.exists():
                    print(
                        f"train mask ablation dataset={dataset} mask={mask_mode} seed={seed}",
                        flush=True,
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "www2027.train_temporal",
                            "--dataset",
                            dataset,
                            "--seed",
                            str(seed),
                            "--mask-mode",
                            mask_mode,
                            "--prior-mode",
                            "none",
                            "--objective",
                            "erm",
                            "--run-name",
                            f"{mask_mode}_anchor_s{seed}",
                            "--method-label",
                            f"{mask_mode}_anchor",
                            "--epochs",
                            "10",
                            "--early-stopping-patience",
                            "3",
                            "--minimum-epochs",
                            "5",
                            "--steps-per-epoch",
                            "50",
                            "--max-eval-batches",
                            "0",
                            "--batch-size",
                            "64",
                            "--max-prefix-length",
                            "50",
                            "--dimension",
                            "32",
                            "--rank",
                            "8",
                            "--context-dim",
                            "8",
                            "--environment-hidden-dim",
                            "16",
                            "--dropout",
                            "0.2",
                            "--learning-rate",
                            "0.001",
                            "--checkpoint-dir",
                            str(args.checkpoint_dir),
                            "--result-json",
                            str(result_path),
                            "--skip-test",
                            "--device",
                            "auto",
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        check=True,
                    )
                elif not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"result exists but checkpoint is missing: {checkpoint_path}"
                    )
                else:
                    print(
                        f"skip mask ablation dataset={dataset} mask={mask_mode} seed={seed}",
                        flush=True,
                    )
                annotate_protocol(result_path)
                completed += 1
    print(f"mask ablation complete runs={completed}")


if __name__ == "__main__":
    main()
