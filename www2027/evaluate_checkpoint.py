"""Evaluate a validation-selected checkpoint on the immutable test split once.

This entry point deliberately does not train or select a model.  If the output
JSON already exists and matches the requested checkpoint, it returns the saved
result instead of touching the test split again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from .data import chronological_split, load_cascades
from .models import TemporalDiffusionModel
from .train_temporal import (
    evaluate_environments_detailed,
    prepare_final_test,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="christian")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--test-environments", type=int, default=3)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="0 evaluates the complete test split",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.checkpoint.resolve()
    result_path = args.result_json.resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)

    # Make repeated invocations idempotent.  This is an audit guard against
    # silently treating the immutable test split as another tuning split.
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("dataset") == args.dataset
        ):
            print(
                f"test_already_evaluated result={result_path} "
                f"map@100={existing['test']['map@100']:.4f}"
            )
            return existing
        raise FileExistsError(
            f"refusing to overwrite an unrelated one-shot test result: {result_path}"
        )

    device = select_device(args.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    records = load_cascades(args.dataset_root / args.dataset)
    split = chronological_split(records)
    observed_num_nodes = max(node for record in records for node in record.cascade) + 1
    if observed_num_nodes != checkpoint["num_nodes"]:
        raise ValueError(
            "checkpoint/data node mismatch: "
            f"{checkpoint['num_nodes']} != {observed_num_nodes}"
        )

    model = TemporalDiffusionModel(
        num_nodes=observed_num_nodes,
        **checkpoint["model_config"],
    ).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    allowed_missing = {
        name for name in missing if name.startswith("temporal_prior_gate.")
    }
    if unexpected or set(missing) != allowed_missing:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, unexpected={unexpected}"
        )
    if checkpoint["model_config"].get("prior_mode") == "temporal" and missing:
        raise RuntimeError("temporal-prior checkpoint is missing its residual head")

    test_environments = [
        environment.graph_to(device)
        for environment in prepare_final_test(
            args,
            observed_num_nodes,
            split.train,
            split.valid,
            split.test,
        )
    ]
    per_environment, test, stratified = evaluate_environments_detailed(
        model,
        test_environments,
        args.batch_size,
        device,
        args.max_eval_batches,
    )
    result: dict[str, object] = {
        "status": "locked_one_shot_test",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "selection_file": (
            str(args.selection_file.resolve()) if args.selection_file else None
        ),
        "selected_epoch": checkpoint["epoch"],
        "saved_validation": checkpoint["validation"],
        "model_config": checkpoint["model_config"],
        "training_objective": {
            "objective": checkpoint["objective"],
            "vrex_weight": checkpoint.get("vrex_weight", 0.0),
            "popularity_balance_alpha": checkpoint.get(
                "popularity_balance_alpha", 0.0
            ),
            "dormant_boost": checkpoint.get("dormant_boost", 0.0),
            "constraint_weight": checkpoint.get("constraint_weight", 0.0),
            "constraint_margin": checkpoint.get("constraint_margin", 0.0),
            "initialization_checkpoint": checkpoint.get("initialization_checkpoint"),
            "freeze_backbone": checkpoint.get("freeze_backbone", False),
            "preservation_weight": checkpoint.get("preservation_weight", 0.0),
            "preservation_topk": checkpoint.get("preservation_topk", 100),
            "preservation_margin": checkpoint.get("preservation_margin", 0.0),
        },
        "protocol": {
            "split": "chronological_70_10_20_timestamp_ties_preserved",
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "valid_environments": args.valid_environments,
            "test_environments": args.test_environments,
            "sample_hop": args.sample_hop,
            "max_eval_batches": args.max_eval_batches,
        },
        "test": test,
        "test_by_environment": per_environment,
        "test_stratified": stratified,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"one_shot_test checkpoint_epoch={checkpoint['epoch']} "
        f"map@100={test['map@100']:.4f} "
        f"worst_map@100={test['worst_map@100']:.4f} result={result_path}"
    )
    return result


def main() -> None:
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()
