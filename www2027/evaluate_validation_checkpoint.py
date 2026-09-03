"""Re-evaluate a selected checkpoint on validation with dual-path reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


@torch.no_grad()
def evaluate_dual_path(
    model: TemporalDiffusionModel,
    environments,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    topk: int,
    protected_cutoffs: tuple[int, ...],
) -> dict[str, object]:
    paths = ("anchor", "adaptive", "fused")
    per_environment = {path: [] for path in paths}
    merged = {
        path: {
            "popularity": {
                name: RankingAccumulator((10, 50, 100))
                for name in POPULARITY_GROUPS
            },
            "recency": {
                name: RankingAccumulator((10, 50, 100))
                for name in RECENCY_GROUPS
            },
        }
        for path in paths
    }
    guarantee_by_cutoff = {
        str(cutoff): {"protected_anchor_hits": 0, "violations": 0}
        for cutoff in protected_cutoffs
    }
    model.eval()
    for environment in environments:
        environment_accumulators = {
            path: RankingAccumulator((10, 50, 100)) for path in paths
        }
        loader = make_loader(
            environment, batch_size, shuffle=False, seed=0
        )
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = batch_to_device(batch, device)
            output = forward_environment(model, environment, batch)
            scores = {
                "anchor": output.base_logits,
                "adaptive": output.logits,
                "fused": protected_union_scores(
                    output.logits,
                    output.base_logits,
                    environment.popularity_groups,
                    environment.recency_groups,
                    topk=topk,
                    protected_cutoffs=protected_cutoffs,
                ),
            }
            target = batch["target"]
            target_popularity = environment.popularity_groups[target]
            target_recency = environment.recency_groups[target]
            for path, logits in scores.items():
                environment_accumulators[path].update(logits, target)
                for index, name in enumerate(POPULARITY_GROUPS):
                    selected = target_popularity == index
                    if bool(torch.any(selected)):
                        merged[path]["popularity"][name].update(
                            logits[selected], target[selected]
                        )
                for index, name in enumerate(RECENCY_GROUPS):
                    selected = target_recency == index
                    if bool(torch.any(selected)):
                        merged[path]["recency"][name].update(
                            logits[selected], target[selected]
                        )

            anchor_target = output.base_logits.gather(1, target.unsqueeze(1))
            fused_target = scores["fused"].gather(1, target.unsqueeze(1))
            anchor_rank = 1 + torch.sum(output.base_logits > anchor_target, dim=1)
            fused_rank = 1 + torch.sum(scores["fused"] > fused_target, dim=1)
            protected = (target_popularity != 0) | (target_recency != 0)
            for cutoff in protected_cutoffs:
                eligible = protected & (anchor_rank <= cutoff)
                guarantee_by_cutoff[str(cutoff)]["protected_anchor_hits"] += int(
                    eligible.sum()
                )
                guarantee_by_cutoff[str(cutoff)]["violations"] += int(
                    (eligible & (fused_rank > cutoff)).sum()
                )

        for path in paths:
            per_environment[path].append(environment_accumulators[path].compute())

    maximum_guarantee = guarantee_by_cutoff[str(max(protected_cutoffs))]
    result: dict[str, object] = {
        "guarantee": {
            "by_cutoff": guarantee_by_cutoff,
            # Preserve the original fields for consumers that audit max K.
            **maximum_guarantee,
        }
    }
    for path in paths:
        strata: dict[str, object] = {}
        for taxonomy, groups in merged[path].items():
            strata[taxonomy] = {}
            for name, accumulator in groups.items():
                values: dict[str, object] = {"count": accumulator.count}
                if accumulator.count:
                    values.update(accumulator.compute())
                strata[taxonomy][name] = values
        result[path] = {
            "by_environment": per_environment[path],
            "metrics": aggregate_environment_metrics(per_environment[path]),
            "stratified": strata,
        }
    return result


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
    parser.add_argument(
        "--rerank-mode",
        choices=("protected_union",),
        default="protected_union",
    )
    parser.add_argument("--rerank-topk", type=int, default=100)
    parser.add_argument("--rerank-cutoffs", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def evaluate_validation_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.checkpoint.resolve()
    result_path = args.result_json.resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    protected_cutoffs = tuple(
        sorted(set(args.rerank_cutoffs or (args.rerank_topk,)))
    )
    rerank_topk = max(protected_cutoffs)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing_cutoffs = existing.get(
            "rerank_cutoffs", [existing.get("rerank_topk")]
        )
        if (
            existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("rerank_mode") == args.rerank_mode
            and existing.get("rerank_topk") == rerank_topk
            and existing_cutoffs == list(protected_cutoffs)
        ):
            print(
                f"validation_rerank_already_evaluated result={result_path} "
                f"map@100={existing['restored_validation']['map@100']:.4f}"
            )
            return existing
        raise FileExistsError(f"refusing to overwrite unrelated result: {result_path}")

    if any(cutoff < 1 for cutoff in protected_cutoffs):
        raise ValueError("rerank cutoffs must be positive")
    device = select_device(args.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    (
        num_nodes,
        _,
        valid_cpu,
        _,
        _,
        _,
    ) = build_prepared_protocol(args)
    if checkpoint["num_nodes"] != num_nodes:
        raise ValueError("checkpoint/data node vocabulary mismatch")

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
    if checkpoint["model_config"].get("prior_mode") != "temporal":
        raise ValueError("protected union requires a temporal-residual checkpoint")

    valid_environments = [environment.graph_to(device) for environment in valid_cpu]
    dual_path = evaluate_dual_path(
        model,
        valid_environments,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_eval_batches,
        topk=rerank_topk,
        protected_cutoffs=protected_cutoffs,
    )
    validation = dual_path["fused"]["metrics"]
    result: dict[str, object] = {
        "status": "validation_only_dual_path_rerank",
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": checkpoint.get("seed"),
        "selected_epoch": checkpoint["epoch"],
        "saved_validation_without_rerank": checkpoint["validation"],
        "rerank_mode": args.rerank_mode,
        "rerank_topk": rerank_topk,
        "rerank_cutoffs": list(protected_cutoffs),
        "restored_validation": validation,
        "validation_by_environment": dual_path["fused"]["by_environment"],
        "validation_stratified": dual_path["fused"]["stratified"],
        "anchor_validation": dual_path["anchor"]["metrics"],
        "anchor_validation_stratified": dual_path["anchor"]["stratified"],
        "adaptive_validation": dual_path["adaptive"]["metrics"],
        "adaptive_validation_stratified": dual_path["adaptive"]["stratified"],
        "guarantee": dual_path["guarantee"],
        "protocol": {
            "split": "chronological_70_10_20_timestamp_ties_preserved",
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "train_environments": args.train_environments,
            "valid_environments": args.valid_environments,
            "sample_hop": args.sample_hop,
            "max_eval_batches": args.max_eval_batches,
            "test_materialized": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"validation_dual_path map@100={validation['map@100']:.4f} "
        f"worst_map@100={validation['worst_map@100']:.4f} result={result_path}"
    )
    return result


def main() -> None:
    evaluate_validation_checkpoint(parse_args())


if __name__ == "__main__":
    main()
