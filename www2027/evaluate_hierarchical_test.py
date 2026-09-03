"""Evaluate one frozen hierarchical dual-path checkpoint on test exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .data import chronological_split, load_cascades
from .evaluate_checkpoint import file_sha256
from .evaluate_validation_checkpoint import evaluate_dual_path
from .models import TemporalDiffusionModel
from .train_temporal import prepare_final_test, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--rerank-cutoffs", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--test-environments", type=int, default=3)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def frozen_entry(
    manifest: dict[str, object],
    *,
    dataset: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> dict[str, object]:
    matches = [
        entry
        for entry in manifest.get("checkpoints", [])
        if entry.get("dataset") == dataset
        and Path(str(entry.get("checkpoint"))).resolve() == checkpoint_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"frozen manifest has {len(matches)} matches for {dataset} {checkpoint_path}"
        )
    entry = matches[0]
    if entry.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs from the frozen manifest")
    return entry


def evaluate_hierarchical_test(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.checkpoint.resolve()
    result_path = args.result_json.resolve()
    selection_path = args.selection_file.resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    selection_sha256 = file_sha256(selection_path)
    cutoffs = tuple(sorted(set(args.rerank_cutoffs)))
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("rerank cutoffs must be positive")

    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_test":
        raise ValueError("selection manifest is not frozen for test")
    if manifest.get("protected_cutoffs") != list(cutoffs):
        raise ValueError("rerank cutoffs differ from the frozen manifest")
    entry = frozen_entry(
        manifest,
        dataset=args.dataset,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )

    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("selection_manifest_sha256") == selection_sha256
            and existing.get("rerank_cutoffs") == list(cutoffs)
        ):
            print(
                f"hierarchical_test_already_evaluated result={result_path} "
                f"map@100={existing['test']['map@100']:.4f}"
            )
            return existing
        raise FileExistsError(
            f"refusing to overwrite unrelated one-shot test result: {result_path}"
        )

    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("seed") is not None and int(checkpoint["seed"]) != int(
        entry["seed"]
    ):
        raise ValueError("checkpoint seed differs from frozen manifest")
    if int(checkpoint["epoch"]) != int(entry["selected_epoch"]):
        raise ValueError("checkpoint epoch differs from frozen manifest")

    records = load_cascades(args.dataset_root / args.dataset)
    split = chronological_split(records)
    num_nodes = max(node for record in records for node in record.cascade) + 1
    if num_nodes != checkpoint["num_nodes"]:
        raise ValueError("checkpoint/data node vocabulary mismatch")

    model = TemporalDiffusionModel(
        num_nodes=num_nodes,
        **checkpoint["model_config"],
    ).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if unexpected or missing:
        raise RuntimeError(
            f"incompatible frozen checkpoint: missing={missing}, unexpected={unexpected}"
        )
    if checkpoint["model_config"].get("prior_mode") != "temporal":
        raise ValueError("hierarchical test requires a temporal-residual checkpoint")

    test_environments = [
        environment.graph_to(device)
        for environment in prepare_final_test(
            args,
            num_nodes,
            split.train,
            split.valid,
            split.test,
        )
    ]
    dual_path = evaluate_dual_path(
        model,
        test_environments,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_eval_batches,
        topk=max(cutoffs),
        protected_cutoffs=cutoffs,
    )
    result: dict[str, object] = {
        "status": "locked_one_shot_hierarchical_test",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": int(entry["seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": checkpoint["epoch"],
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha256,
        "rerank_mode": "hierarchical_protected_union",
        "rerank_cutoffs": list(cutoffs),
        "test": dual_path["fused"]["metrics"],
        "test_by_environment": dual_path["fused"]["by_environment"],
        "test_stratified": dual_path["fused"]["stratified"],
        "anchor_test": dual_path["anchor"]["metrics"],
        "anchor_test_stratified": dual_path["anchor"]["stratified"],
        "adaptive_test": dual_path["adaptive"]["metrics"],
        "adaptive_test_stratified": dual_path["adaptive"]["stratified"],
        "guarantee": dual_path["guarantee"],
        "protocol": {
            "split": "chronological_70_10_20_timestamp_ties_preserved",
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "valid_environments": args.valid_environments,
            "test_environments": args.test_environments,
            "sample_hop": args.sample_hop,
            "max_eval_batches": args.max_eval_batches,
            "test_materialized": True,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"one_shot_hierarchical_test dataset={args.dataset} seed={entry['seed']} "
        f"map@100={result['test']['map@100']:.4f} "
        f"worst_map@100={result['test']['worst_map@100']:.4f} "
        f"result={result_path}"
    )
    return result


def main() -> None:
    evaluate_hierarchical_test(parse_args())


if __name__ == "__main__":
    main()
