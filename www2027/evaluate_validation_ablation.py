"""Evaluate the frozen inference chain ablations on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch

from .evaluate_checkpoint import file_sha256
from .metrics import (
    POPULARITY_GROUPS,
    RECENCY_GROUPS,
    RankingAccumulator,
    aggregate_environment_metrics,
    protected_union_scores,
)
from .models import TemporalDiffusionModel
from .train_temporal import (
    batch_to_device,
    build_prepared_protocol,
    forward_environment,
    make_loader,
    select_device,
)


PATHS = ("anchor", "adaptive", "top100_union", "hierarchical_union")
CUTOFFS = (10, 50, 100)


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def checkpoint_seed(path: Path, checkpoint: dict[str, object]) -> int:
    if checkpoint.get("seed") is not None:
        return int(checkpoint["seed"])
    match = re.search(r"_s(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot recover seed from checkpoint {path}")
    return int(match.group(1))


def make_stratified_accumulators() -> dict[str, dict[str, RankingAccumulator]]:
    return {
        "popularity": {
            name: RankingAccumulator(CUTOFFS) for name in POPULARITY_GROUPS
        },
        "recency": {
            name: RankingAccumulator(CUTOFFS) for name in RECENCY_GROUPS
        },
    }


def finish_stratified(
    accumulators: dict[str, dict[str, RankingAccumulator]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for taxonomy, groups in accumulators.items():
        result[taxonomy] = {}
        for name, accumulator in groups.items():
            values: dict[str, object] = {"count": accumulator.count}
            if accumulator.count:
                values.update(accumulator.compute())
            result[taxonomy][name] = values
    return result


@torch.no_grad()
def evaluate(
    model: TemporalDiffusionModel,
    environments,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, object]:
    per_environment = {path: [] for path in PATHS}
    stratified = {path: make_stratified_accumulators() for path in PATHS}
    guarantee = {
        path: {
            str(cutoff): {"protected_anchor_hits": 0, "violations": 0}
            for cutoff in CUTOFFS
        }
        for path in ("top100_union", "hierarchical_union")
    }

    model.eval()
    for environment in environments:
        environment_accumulators = {
            path: RankingAccumulator(CUTOFFS) for path in PATHS
        }
        loader = make_loader(environment, batch_size, shuffle=False, seed=0)
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = batch_to_device(batch, device)
            output = forward_environment(model, environment, batch)
            scores = {
                "anchor": output.base_logits,
                "adaptive": output.logits,
                "top100_union": protected_union_scores(
                    output.logits,
                    output.base_logits,
                    environment.popularity_groups,
                    environment.recency_groups,
                    topk=100,
                    protected_cutoffs=(100,),
                ),
                "hierarchical_union": protected_union_scores(
                    output.logits,
                    output.base_logits,
                    environment.popularity_groups,
                    environment.recency_groups,
                    topk=100,
                    protected_cutoffs=CUTOFFS,
                ),
            }
            target = batch["target"]
            target_popularity = environment.popularity_groups[target]
            target_recency = environment.recency_groups[target]
            protected = (target_popularity != 0) | (target_recency != 0)

            for path, logits in scores.items():
                environment_accumulators[path].update(logits, target)
                for index, name in enumerate(POPULARITY_GROUPS):
                    selected = target_popularity == index
                    if bool(torch.any(selected)):
                        stratified[path]["popularity"][name].update(
                            logits[selected], target[selected]
                        )
                for index, name in enumerate(RECENCY_GROUPS):
                    selected = target_recency == index
                    if bool(torch.any(selected)):
                        stratified[path]["recency"][name].update(
                            logits[selected], target[selected]
                        )

            anchor_target = output.base_logits.gather(1, target.unsqueeze(1))
            anchor_rank = 1 + torch.sum(output.base_logits > anchor_target, dim=1)
            for path in guarantee:
                fused_target = scores[path].gather(1, target.unsqueeze(1))
                fused_rank = 1 + torch.sum(scores[path] > fused_target, dim=1)
                for cutoff in CUTOFFS:
                    eligible = protected & (anchor_rank <= cutoff)
                    item = guarantee[path][str(cutoff)]
                    item["protected_anchor_hits"] += int(eligible.sum())
                    item["violations"] += int(
                        (eligible & (fused_rank > cutoff)).sum()
                    )

        for path in PATHS:
            per_environment[path].append(
                environment_accumulators[path].compute()
            )

    return {
        "paths": {
            path: {
                "metrics": aggregate_environment_metrics(per_environment[path]),
                "by_environment": per_environment[path],
                "stratified": finish_stratified(stratified[path]),
            }
            for path in PATHS
        },
        "guarantee": guarantee,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    result_path = args.result_json.resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("checkpoint_sha256") == checkpoint_hash
            and existing.get("paths") is not None
            and existing.get("protocol", {}).get("test_materialized") is False
        ):
            print(f"validation_ablation_already_evaluated result={result_path}")
            return
        raise FileExistsError(f"refusing to overwrite unrelated result: {result_path}")

    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_nodes, _, valid_cpu, _, _, _ = build_prepared_protocol(args)
    if checkpoint["num_nodes"] != num_nodes:
        raise ValueError("checkpoint/data node vocabulary mismatch")
    if checkpoint["model_config"].get("prior_mode") != "temporal":
        raise ValueError("inference ablation requires a temporal-residual checkpoint")

    model = TemporalDiffusionModel(
        num_nodes=num_nodes,
        **checkpoint["model_config"],
    ).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if unexpected or any(
        not name.startswith("temporal_prior_gate.") for name in missing
    ):
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, unexpected={unexpected}"
        )
    validation = evaluate(
        model,
        [environment.graph_to(device) for environment in valid_cpu],
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_eval_batches,
    )
    result = {
        "status": "postfreeze_validation_inference_ablation",
        "dataset": args.dataset,
        "seed": checkpoint_seed(checkpoint_path, checkpoint),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": checkpoint["epoch"],
        **validation,
        "protocol": {
            "split": "chronological_70_10_20_timestamp_ties_preserved",
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
        f"validation_ablation dataset={args.dataset} seed={result['seed']} "
        f"result={result_path}"
    )


if __name__ == "__main__":
    main()
