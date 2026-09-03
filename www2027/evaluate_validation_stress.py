"""Evaluate frozen checkpoints under past-only validation popularity shocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch

from .evaluate_checkpoint import file_sha256
from .evaluate_validation_checkpoint import evaluate_dual_path
from .models import TemporalDiffusionModel
from .stress import STRESSES, perturb_environment
from .train_temporal import build_prepared_protocol, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--stresses", nargs="+", choices=STRESSES, default=list(STRESSES))
    parser.add_argument("--severities", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--rerank-cutoffs", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _seed_from_checkpoint(path: Path) -> int:
    match = re.search(r"_s(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot recover seed from checkpoint name: {path.name}")
    return int(match.group(1))


def _metric_delta(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    return {key: float(left[key]) - float(right[key]) for key in left}


def evaluate_validation_stress(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.checkpoint.resolve()
    result_path = args.result_json.resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    stresses = tuple(dict.fromkeys(args.stresses))
    severities = tuple(sorted(set(args.severities)))
    cutoffs = tuple(sorted(set(args.rerank_cutoffs)))
    if any(not 0.0 < severity <= 1.0 for severity in severities):
        raise ValueError("stress severities must be in (0, 1]")
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("rerank cutoffs must be positive")

    signature = {
        "checkpoint_sha256": checkpoint_sha256,
        "stresses": list(stresses),
        "severities": list(severities),
        "rerank_cutoffs": list(cutoffs),
    }
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            print(f"validation_stress_already_evaluated result={result_path}")
            return existing
        raise FileExistsError(f"refusing to overwrite unrelated result: {result_path}")

    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_nodes, _, valid_cpu, _, _, _ = build_prepared_protocol(args)
    if checkpoint["num_nodes"] != num_nodes:
        raise ValueError("checkpoint/data node vocabulary mismatch")
    model = TemporalDiffusionModel(
        num_nodes=num_nodes,
        **checkpoint["model_config"],
    ).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, unexpected={unexpected}"
        )
    if checkpoint["model_config"].get("prior_mode") != "temporal":
        raise ValueError("stress evaluation requires a temporal-residual checkpoint")

    valid = [environment.graph_to(device) for environment in valid_cpu]
    evaluation_args = {
        "batch_size": args.batch_size,
        "device": device,
        "max_batches": args.max_eval_batches,
        "topk": max(cutoffs),
        "protected_cutoffs": cutoffs,
    }
    baseline = evaluate_dual_path(model, valid, **evaluation_args)
    baseline_fused = baseline["fused"]["metrics"]
    conditions: list[dict[str, object]] = []
    for stress in stresses:
        for severity in severities:
            perturbed = [
                perturb_environment(environment, stress, severity)
                for environment in valid
            ]
            evaluated = evaluate_dual_path(model, perturbed, **evaluation_args)
            fused = evaluated["fused"]["metrics"]
            anchor = evaluated["anchor"]["metrics"]
            conditions.append(
                {
                    "stress": stress,
                    "severity": severity,
                    "anchor": anchor,
                    "adaptive": evaluated["adaptive"]["metrics"],
                    "fused": fused,
                    "fused_minus_anchor": _metric_delta(fused, anchor),
                    "fused_minus_unperturbed": _metric_delta(fused, baseline_fused),
                    "guarantee": evaluated["guarantee"],
                    "fused_stratified": evaluated["fused"]["stratified"],
                }
            )

    result: dict[str, object] = {
        "status": "post_freeze_validation_only_descriptive_stress",
        "dataset": args.dataset,
        "seed": _seed_from_checkpoint(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        **signature,
        "selected_epoch": checkpoint["epoch"],
        "perturbation_scope": (
            "past-only recent popularity and recomputed safety groups; "
            "graph, environment context, cumulative history, and targets fixed"
        ),
        "baseline": {
            "anchor": baseline["anchor"]["metrics"],
            "adaptive": baseline["adaptive"]["metrics"],
            "fused": baseline_fused,
            "guarantee": baseline["guarantee"],
        },
        "conditions": conditions,
        "protocol": {
            "split": "chronological_validation_only",
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "train_environments": args.train_environments,
            "valid_environments": args.valid_environments,
            "sample_hop": args.sample_hop,
            "max_eval_batches": args.max_eval_batches,
            "test_materialized": False,
            "selection_changes_permitted": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"validation_stress dataset={args.dataset} seed={result['seed']} "
        f"conditions={len(conditions)} result={result_path}"
    )
    return result


def main() -> None:
    evaluate_validation_stress(parse_args())


if __name__ == "__main__":
    main()
