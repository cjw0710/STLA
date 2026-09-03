"""Train an internal environment-conditioned low-rank correction for DeDiff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .metrics import RankingAccumulator, aggregate_environment_metrics, protected_union_scores
from .models import DynamicDeDiff
from .train_dediff_logit_adapter import (
    PATHS,
    _anchor_arguments,
    _audit_anchor,
    build_contexts,
    load_frozen_anchor,
)
from .train_temporal_dediff import (
    CUTOFFS,
    DeDiff,
    DeDiffEnvironment,
    make_loaders,
    prepare_protocol,
)
from .training import PreparedEnvironment


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--temporal-rank", type=int, default=8)
    parser.add_argument("--base-rank", type=int, default=0)
    parser.add_argument("--temporal-hidden-dim", type=int, default=32)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _candidate_logits(prediction: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [prediction.new_full((prediction.shape[0], 1), float("-inf")), prediction[:, 1:-1]],
        dim=1,
    )


def _forward_anchor(
    anchor: DeDiff,
    model_args,
    batch: dict[str, torch.Tensor],
    environment: DeDiffEnvironment,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        prediction, label, _ = anchor(model_args, batch, environment.info)
    return _candidate_logits(prediction), label


def _forward_dynamic(
    dynamic: DynamicDeDiff,
    model_args,
    batch: dict[str, torch.Tensor],
    environment: DeDiffEnvironment,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, label, _ = dynamic(model_args, batch, environment.info)
    return _candidate_logits(prediction), label


def load_dynamic_model(
    anchor_result: dict[str, Any],
    model_args,
    args: argparse.Namespace,
) -> DynamicDeDiff:
    model = DynamicDeDiff(
        model_args,
        temporal_rank=args.temporal_rank,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_dropout=args.temporal_dropout,
    ).to(model_args.device)
    checkpoint = torch.load(
        Path(anchor_result["checkpoint"]),
        map_location=model_args.device,
        weights_only=False,
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected anchor keys: {incompatible.unexpected_keys}")
    if not incompatible.missing_keys or any(
        not key.startswith("temporal_debias.") for key in incompatible.missing_keys
    ):
        raise ValueError(f"invalid temporal parameter initialization: {incompatible.missing_keys}")
    model.freeze_anchor()
    if args.base_rank:
        model.compress_debiasing(args.base_rank)
    return model


def attach_contexts(
    environments: Sequence[DeDiffEnvironment],
    contexts: Sequence[PreparedEnvironment],
    device: torch.device,
) -> list[PreparedEnvironment]:
    moved = [context.graph_to(device) for context in contexts]
    if [item.name for item in environments] != [item.name for item in moved]:
        raise RuntimeError("DeDiff and temporal context environments are misaligned")
    for environment, context in zip(environments, moved):
        environment.info["environment_features"] = context.environment_features
        environment.info["A_interaction_dynamic"] = (
            environment.info["A_interaction"].to_sparse_coo().coalesce()
        )
        environment.info["A_social_dynamic"] = (
            environment.info["A_social"].to_sparse_coo().coalesce()
        )
    return moved


@torch.no_grad()
def audit_zero_correction(
    anchor: DeDiff,
    dynamic: DynamicDeDiff,
    model_args,
    environment: DeDiffEnvironment,
    loader: DataLoader,
) -> dict[str, float]:
    anchor.eval()
    dynamic.eval()
    batch = next(iter(loader))
    anchor_logits, label = _forward_anchor(anchor, model_args, batch, environment)
    dynamic_logits, _ = _forward_dynamic(dynamic, model_args, batch, environment)
    valid = label.ne(0) & label.ne(model_args.user_num - 1)
    anchor_real = anchor_logits[valid, 1:]
    dynamic_real = dynamic_logits[valid, 1:]
    finite = torch.isfinite(anchor_real) & torch.isfinite(dynamic_real)
    finite_difference = (anchor_real[finite] - dynamic_real[finite]).abs()
    anchor_top = torch.topk(anchor_logits[valid], k=100, dim=1).indices
    dynamic_top = torch.topk(dynamic_logits[valid], k=100, dim=1).indices
    return {
        "max_abs_logit_difference": float(finite_difference.max()),
        "mean_abs_logit_difference": float(finite_difference.mean()),
        "top100_exact_row_fraction": float(torch.all(anchor_top == dynamic_top, dim=1).float().mean()),
    }


def train_epoch(
    dynamic: DynamicDeDiff,
    model_args,
    environments: Sequence[DeDiffEnvironment],
    loaders: Sequence[DataLoader],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    gradient_clip: float,
) -> float:
    # Keep the frozen DeDiff backbone, including BatchNorm and dropout, in
    # inference mode while training only the temporal correction.
    dynamic.eval()
    dynamic.temporal_debias.train()
    iterators = [iter(loader) for loader in loaders]
    total_loss = 0.0
    parameters = list(dynamic.temporal_parameters())
    for step in range(steps):
        environment_index = step % len(loaders)
        try:
            batch = next(iterators[environment_index])
        except StopIteration:
            iterators[environment_index] = iter(loaders[environment_index])
            batch = next(iterators[environment_index])
        adaptive_logits, label = _forward_dynamic(
            dynamic,
            model_args,
            batch,
            environments[environment_index],
        )
        valid = label.ne(0) & label.ne(model_args.user_num - 1)
        loss = torch.nn.functional.cross_entropy(adaptive_logits[valid], label[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(parameters, gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / steps


@torch.no_grad()
def evaluate(
    anchor: DeDiff,
    dynamic: DynamicDeDiff,
    model_args,
    environments: Sequence[DeDiffEnvironment],
    contexts: Sequence[PreparedEnvironment],
    loaders: Sequence[DataLoader],
    *,
    max_batches: int,
    exact_anchor_fallback: bool = False,
) -> dict[str, Any]:
    anchor.eval()
    dynamic.eval()
    by_path = {path: [] for path in PATHS}
    guarantee = {
        str(cutoff): {"protected_anchor_hits": 0, "violations": 0}
        for cutoff in CUTOFFS
    }
    for environment, context, loader in zip(environments, contexts, loaders):
        accumulators = {path: RankingAccumulator(CUTOFFS) for path in PATHS}
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            anchor_all, label = _forward_anchor(anchor, model_args, batch, environment)
            if exact_anchor_fallback:
                adaptive_all = anchor_all
            else:
                adaptive_all, _ = _forward_dynamic(dynamic, model_args, batch, environment)
            valid = label.ne(0) & label.ne(model_args.user_num - 1)
            target = label[valid]
            anchor_logits = anchor_all[valid]
            adaptive_logits = adaptive_all[valid]
            fused_logits = protected_union_scores(
                adaptive_logits,
                anchor_logits,
                context.popularity_groups,
                context.recency_groups,
                topk=100,
                protected_cutoffs=CUTOFFS,
            )
            scores = {
                "anchor": anchor_logits,
                "adaptive": adaptive_logits,
                "hierarchical_union": fused_logits,
            }
            for path, logits in scores.items():
                accumulators[path].update(logits, target)

            protected = (
                (context.popularity_groups[target] != 0)
                | (context.recency_groups[target] != 0)
            )
            anchor_score = anchor_logits.gather(1, target.unsqueeze(1))
            anchor_rank = 1 + torch.sum(anchor_logits > anchor_score, dim=1)
            fused_score = fused_logits.gather(1, target.unsqueeze(1))
            fused_rank = 1 + torch.sum(fused_logits > fused_score, dim=1)
            for cutoff in CUTOFFS:
                eligible = protected & (anchor_rank <= cutoff)
                guarantee[str(cutoff)]["protected_anchor_hits"] += int(eligible.sum())
                guarantee[str(cutoff)]["violations"] += int(
                    (eligible & (fused_rank > cutoff)).sum()
                )
        for path in PATHS:
            by_path[path].append(accumulators[path].compute())
    return {
        "paths": {
            path: {
                "metrics": aggregate_environment_metrics(values),
                "by_environment": values,
            }
            for path, values in by_path.items()
        },
        "guarantee": guarantee,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the preserved DeDiff backbone requires CUDA")
    anchor_result = json.loads(args.anchor_result.read_text(encoding="utf-8"))
    _audit_anchor(anchor_result)
    seed = int(anchor_result["seed"])
    seed_everything(seed)
    anchor_args = _anchor_arguments(anchor_result, args.dataset_root)
    model_args, train_environments, valid_environments, _, num_nodes = prepare_protocol(anchor_args)
    train_contexts, valid_contexts = build_contexts(anchor_args, num_nodes=num_nodes)
    train_contexts = attach_contexts(train_environments, train_contexts, model_args.device)
    valid_contexts = attach_contexts(valid_environments, valid_contexts, model_args.device)
    train_loaders = make_loaders(
        train_environments,
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
    )
    valid_loaders = make_loaders(
        valid_environments,
        batch_size=args.batch_size,
        shuffle=False,
        seed=seed,
    )
    anchor = load_frozen_anchor(anchor_result, anchor_args, model_args)
    dynamic = load_dynamic_model(anchor_result, model_args, args)
    equivalence = audit_zero_correction(
        anchor,
        dynamic,
        model_args,
        valid_environments[0],
        valid_loaders[0],
    )
    print(json.dumps({"zero_correction_audit": equivalence}, indent=2), flush=True)
    optimizer = torch.optim.Adam(
        dynamic.temporal_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)

    initial = evaluate(
        anchor,
        dynamic,
        model_args,
        valid_environments,
        valid_contexts,
        valid_loaders,
        max_batches=args.max_eval_batches,
        exact_anchor_fallback=True,
    )
    best_score = initial["paths"]["hierarchical_union"]["metrics"]["map@100"]
    selected_epoch = 0
    torch.save(
        {"temporal_state": dynamic.temporal_debias.state_dict(), "selected_epoch": 0},
        args.checkpoint,
    )
    history: list[dict[str, Any]] = [{"epoch": 0, "training_loss": None, "evaluation": initial}]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            dynamic,
            model_args,
            train_environments,
            train_loaders,
            optimizer,
            steps=args.steps_per_epoch,
            gradient_clip=args.gradient_clip,
        )
        evaluation = evaluate(
            anchor,
            dynamic,
            model_args,
            valid_environments,
            valid_contexts,
            valid_loaders,
            max_batches=args.max_eval_batches,
        )
        score = evaluation["paths"]["hierarchical_union"]["metrics"]["map@100"]
        history.append({"epoch": epoch, "training_loss": loss, "evaluation": evaluation})
        print(f"epoch={epoch:03d} loss={loss:.6f} fused_map@100={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            stale = 0
            torch.save(
                {"temporal_state": dynamic.temporal_debias.state_dict(), "selected_epoch": epoch},
                args.checkpoint,
            )
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    selected = torch.load(args.checkpoint, map_location=model_args.device, weights_only=False)
    dynamic.temporal_debias.load_state_dict(selected["temporal_state"])
    final = evaluate(
        anchor,
        dynamic,
        model_args,
        valid_environments,
        valid_contexts,
        valid_loaders,
        max_batches=args.max_eval_batches,
        exact_anchor_fallback=selected_epoch == 0,
    )
    result = {
        "status": "validation_only_dynamic_internal_dediff",
        "dataset": anchor_result["dataset"],
        "seed": seed,
        "anchor_result": str(args.anchor_result.resolve()),
        "anchor_checkpoint": anchor_result["checkpoint"],
        "selected_epoch": selected_epoch,
        "temporal_rank": args.temporal_rank,
        "base_rank": dynamic.compressed_debias_rank,
        "base_spectral_energy_retained": dynamic.compressed_debias_energy,
        "base_factor_float_count": (
            0
            if dynamic.compressed_debias_rank == 0
            else 2 * model_args.user_num * dynamic.compressed_debias_rank
        ),
        "temporal_parameter_count": sum(
            parameter.numel() for parameter in dynamic.temporal_parameters()
        ),
        "zero_correction_equivalence": equivalence,
        "evaluation": final,
        "history": history,
        "protocol": {
            "relationship_to_cikm": "internal low-rank correction of DeDiff debiasing operator",
            "associative_rewrite": "(A @ D) @ X -> A @ (D @ X)",
            "dynamic_graph_multiplication_sparse": True,
            "dense_debiasing_replaced": dynamic.compressed_debias_rank > 0,
            "anchor_frozen": True,
            "past_only_environment_conditioning": True,
            "chronological_split": [0.7, 0.1, 0.2],
            "selection_metric": "hierarchical union mean validation MAP@100",
            "exact_epoch_zero_fallback": True,
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "postfreeze_descriptive": True,
        },
    }
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "temporal_parameter_count": result["temporal_parameter_count"],
                "metrics": final["paths"]["hierarchical_union"]["metrics"],
                "guarantee": final["guarantee"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
