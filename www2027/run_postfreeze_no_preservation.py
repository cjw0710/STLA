"""Run the validation-only no-preservation-loss ablation from frozen bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "pretest_selection_manifest.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["christian", "android", "douban", "twitter"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[21, 42, 84, 126, 168]
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints"
        / "postfreeze_no_preservation",
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "postfreeze_no_preservation_training",
    )
    parser.add_argument(
        "--evaluation-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "postfreeze_no_preservation_hierarchical",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def annotate_postfreeze_protocol(path: Path) -> None:
    """Backfill protocol metadata for results written by the legacy trainer."""

    item = json.loads(path.read_text(encoding="utf-8"))
    protocol = item.setdefault(
        "protocol",
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
        },
    )
    protocol["test_materialized"] = False
    protocol["selection_changes_permitted"] = False
    protocol["postfreeze_descriptive_ablation"] = True
    path.write_text(json.dumps(item, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    learning_rates = manifest["residual_learning_rate"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.training_output_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        entry
        for entry in manifest["checkpoints"]
        if entry["dataset"] in args.datasets and int(entry["seed"]) in args.seeds
    ]
    for entry in selected:
        dataset, seed = str(entry["dataset"]), int(entry["seed"])
        final_checkpoint = torch.load(
            entry["checkpoint"], map_location="cpu", weights_only=False
        )
        config = final_checkpoint["model_config"]
        initialization = Path(str(final_checkpoint["initialization_checkpoint"]))
        if not initialization.exists():
            raise FileNotFoundError(initialization)

        training_result = args.training_output_dir / f"{dataset}_s{seed}.json"
        ablation_checkpoint = (
            args.checkpoint_dir / f"{dataset}_no_preservation_s{seed}.pt"
        )
        if not training_result.exists():
            print(f"train no-preservation dataset={dataset} seed={seed}", flush=True)
            run(
                [
                    sys.executable,
                    "-m",
                    "www2027.train_temporal",
                    "--dataset",
                    dataset,
                    "--seed",
                    str(seed),
                    "--mask-mode",
                    str(config["mask_mode"]),
                    "--prior-mode",
                    "temporal",
                    "--objective",
                    "erm",
                    "--run-name",
                    f"no_preservation_s{seed}",
                    "--method-label",
                    "no_preservation",
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
                    str(config["dimension"]),
                    "--rank",
                    str(config["rank"]),
                    "--context-dim",
                    str(config["context_dim"]),
                    "--environment-hidden-dim",
                    str(config["environment_hidden_dim"]),
                    "--dropout",
                    str(config["dropout"]),
                    "--learning-rate",
                    str(learning_rates[dataset]),
                    "--initialize-checkpoint",
                    str(initialization),
                    "--freeze-backbone",
                    "--preservation-weight",
                    "0",
                    "--preservation-topk",
                    "100",
                    "--preservation-margin",
                    "0",
                    "--checkpoint-dir",
                    str(args.checkpoint_dir),
                    "--result-json",
                    str(training_result),
                    "--skip-test",
                    "--device",
                    "auto",
                ]
            )
        elif not ablation_checkpoint.exists():
            raise FileNotFoundError(
                f"training result exists but checkpoint is missing: {ablation_checkpoint}"
            )
        else:
            print(f"skip training dataset={dataset} seed={seed}", flush=True)
        annotate_postfreeze_protocol(training_result)

        evaluation_result = args.evaluation_output_dir / f"{dataset}_s{seed}.json"
        if not evaluation_result.exists():
            print(f"evaluate no-preservation dataset={dataset} seed={seed}", flush=True)
            run(
                [
                    sys.executable,
                    "-m",
                    "www2027.evaluate_validation_checkpoint",
                    "--dataset",
                    dataset,
                    "--checkpoint",
                    str(ablation_checkpoint),
                    "--result-json",
                    str(evaluation_result),
                    "--rerank-cutoffs",
                    "10",
                    "50",
                    "100",
                    "--batch-size",
                    "64",
                    "--max-prefix-length",
                    "50",
                    "--device",
                    "auto",
                ]
            )
        else:
            print(f"skip evaluation dataset={dataset} seed={seed}", flush=True)

    print(f"no-preservation complete runs={len(selected)}")


if __name__ == "__main__":
    main()
