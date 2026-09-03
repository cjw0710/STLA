"""Train DriftDiff with chronological environments and validation-only selection."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import (
    CascadeRecord,
    TemporalEnvironment,
    build_rolling_snapshots,
    chronological_split,
    load_cascades,
    make_temporal_environments,
)
from .metrics import (
    POPULARITY_GROUPS,
    RECENCY_GROUPS,
    RankingAccumulator,
    aggregate_environment_metrics,
    popularity_group_ids,
    protected_union_scores,
    recency_group_ids,
)
from .models import TemporalDiffusionModel
from .training import (
    ERM,
    GroupDRO,
    PreparedEnvironment,
    VREx,
    environment_loss,
    prepare_environment,
    topk_subgroup_preservation_penalty,
)


@dataclass
class EpochSummary:
    epoch: int
    training_loss: float
    group_weights: list[float]
    environment_prediction_loss: list[float]
    environment_constraint_penalty: list[float]
    environment_preservation_penalty: list[float]
    validation: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="christian")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=100)
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--test-environments", type=int, default=3)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--environment-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--mask-mode",
        choices=("dynamic", "static", "none"),
        default="dynamic",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("none", "temporal"),
        default="none",
        help="Optional sample-conditioned residual from past-only node state",
    )
    parser.add_argument(
        "--objective",
        choices=("groupdro", "erm", "vrex"),
        default="erm",
    )
    parser.add_argument("--vrex-weight", type=float, default=1.0)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--group-dro-step-size", type=float, default=0.01)
    parser.add_argument("--shortcut-weight", type=float, default=0.1)
    parser.add_argument("--independence-weight", type=float, default=0.05)
    parser.add_argument("--mask-balance-weight", type=float, default=0.01)
    parser.add_argument("--popularity-balance-alpha", type=float, default=0.0)
    parser.add_argument("--dormant-boost", type=float, default=0.0)
    parser.add_argument("--constraint-weight", type=float, default=0.0)
    parser.add_argument("--constraint-margin", type=float, default=0.5)
    parser.add_argument(
        "--initialize-checkpoint",
        type=Path,
        help="Optional validation-selected base checkpoint used for residual tuning",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the temporal residual head after checkpoint initialization",
    )
    parser.add_argument("--preservation-weight", type=float, default=0.0)
    parser.add_argument("--preservation-topk", type=int, default=100)
    parser.add_argument("--preservation-margin", type=float, default=0.0)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after restored validation metrics; never materialize test data",
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="0 uses the longest environment loader; set 1 for a smoke test",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="0 evaluates every batch",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    parser.add_argument(
        "--run-name",
        default="driftdiff",
        help="Safe suffix used to keep checkpoints from different runs separate",
    )
    parser.add_argument(
        "--method-label",
        default="",
        help="Optional experiment label written to result JSON",
    )
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_prepared_protocol(
    args: argparse.Namespace,
) -> tuple[
    int,
    list[PreparedEnvironment],
    list[PreparedEnvironment],
    tuple[CascadeRecord, ...],
    tuple[CascadeRecord, ...],
    tuple[CascadeRecord, ...],
]:
    records = load_cascades(args.dataset_root / args.dataset)
    num_nodes = max(node for record in records for node in record.cascade) + 1
    split = chronological_split(records)

    train_environments = make_temporal_environments(
        split.train, args.train_environments, prefix="train"
    )
    valid_environments = make_temporal_environments(
        split.valid, args.valid_environments, prefix="valid"
    )
    train_snapshots = build_rolling_snapshots(
        train_environments,
        num_nodes,
        sample_hop=args.sample_hop,
    )
    valid_snapshots = build_rolling_snapshots(
        valid_environments,
        num_nodes,
        warm_start_records=split.train,
        warm_start_recent_records=train_environments[-1].records,
        sample_hop=args.sample_hop,
    )
    def prepare(snapshots: Sequence) -> list[PreparedEnvironment]:
        return [
            prepare_environment(snapshot, num_nodes, args.max_prefix_length)
            for snapshot in snapshots
        ]

    # Test cascades are retained as immutable records but are not converted to
    # graphs, features, datasets, or device tensors before model selection.
    return (
        num_nodes,
        prepare(train_snapshots),
        prepare(valid_snapshots),
        split.train,
        split.valid,
        split.test,
    )


def prepare_final_test(
    args: argparse.Namespace,
    num_nodes: int,
    train_records: tuple[CascadeRecord, ...],
    valid_records: tuple[CascadeRecord, ...],
    test_records: tuple[CascadeRecord, ...],
) -> list[PreparedEnvironment]:
    """Materialize test environments only after checkpoint selection."""

    environments = make_temporal_environments(
        test_records, args.test_environments, prefix="test"
    )
    snapshots = build_rolling_snapshots(
        environments,
        num_nodes,
        warm_start_records=(*train_records, *valid_records),
        warm_start_recent_records=make_temporal_environments(
            valid_records, args.valid_environments, prefix="valid"
        )[-1].records,
        sample_hop=args.sample_hop,
    )
    return [
        prepare_environment(snapshot, num_nodes, args.max_prefix_length)
        for snapshot in snapshots
    ]


def make_loader(
    environment: PreparedEnvironment,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if len(environment.dataset) == 0:
        raise ValueError(f"environment {environment.name} has no next-user examples")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        environment.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_environment(
    model: TemporalDiffusionModel,
    environment: PreparedEnvironment,
    batch: dict[str, torch.Tensor],
):
    return model(
        prefix=batch["prefix"],
        elapsed=batch["elapsed"],
        lengths=batch["length"],
        edge_index=environment.edge_index,
        edge_weight=environment.edge_weight,
        environment_features=environment.environment_features,
        historical_popularity=environment.historical_popularity,
        recent_popularity=environment.recent_popularity,
    )


@torch.no_grad()
def evaluate_environment_accumulators(
    model: TemporalDiffusionModel,
    environment: PreparedEnvironment,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    *,
    stratified: bool,
    rerank_mode: str = "none",
    rerank_topk: int = 100,
) -> tuple[RankingAccumulator, dict[str, dict[str, RankingAccumulator]]]:
    if rerank_mode not in {"none", "protected_union"}:
        raise ValueError("rerank mode must be none or protected_union")
    model.eval()
    accumulator = RankingAccumulator((10, 50, 100))
    strata: dict[str, dict[str, RankingAccumulator]] = {}
    popularity_ids: torch.Tensor | None = None
    recency_ids: torch.Tensor | None = None
    if stratified:
        strata = {
            "popularity": {
                name: RankingAccumulator((10, 50, 100)) for name in POPULARITY_GROUPS
            },
            "recency": {
                name: RankingAccumulator((10, 50, 100)) for name in RECENCY_GROUPS
            },
        }
        popularity_ids = popularity_group_ids(environment.historical_popularity)
        recency_ids = recency_group_ids(
            environment.historical_popularity,
            environment.recent_popularity,
        )
    loader = make_loader(environment, batch_size, shuffle=False, seed=0)
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = batch_to_device(batch, device)
        output = forward_environment(model, environment, batch)
        ranking_logits = output.logits
        if rerank_mode == "protected_union":
            ranking_logits = protected_union_scores(
                output.logits,
                output.base_logits,
                environment.popularity_groups,
                environment.recency_groups,
                topk=rerank_topk,
            )
        accumulator.update(ranking_logits, batch["target"])
        if stratified:
            assert popularity_ids is not None and recency_ids is not None
            for group_index, name in enumerate(POPULARITY_GROUPS):
                selected = popularity_ids[batch["target"]] == group_index
                if bool(torch.any(selected)):
                    strata["popularity"][name].update(
                        ranking_logits[selected], batch["target"][selected]
                    )
            for group_index, name in enumerate(RECENCY_GROUPS):
                selected = recency_ids[batch["target"]] == group_index
                if bool(torch.any(selected)):
                    strata["recency"][name].update(
                        ranking_logits[selected], batch["target"][selected]
                    )
    return accumulator, strata


@torch.no_grad()
def evaluate_environment(
    model: TemporalDiffusionModel,
    environment: PreparedEnvironment,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    *,
    rerank_mode: str = "none",
    rerank_topk: int = 100,
) -> dict[str, float]:
    accumulator, _ = evaluate_environment_accumulators(
        model,
        environment,
        batch_size,
        device,
        max_batches,
        stratified=False,
        rerank_mode=rerank_mode,
        rerank_topk=rerank_topk,
    )
    return accumulator.compute()


def evaluate_environments(
    model: TemporalDiffusionModel,
    environments: Sequence[PreparedEnvironment],
    batch_size: int,
    device: torch.device,
    max_batches: int,
    *,
    rerank_mode: str = "none",
    rerank_topk: int = 100,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    per_environment = [
        evaluate_environment(
            model,
            environment,
            batch_size,
            device,
            max_batches,
            rerank_mode=rerank_mode,
            rerank_topk=rerank_topk,
        )
        for environment in environments
    ]
    return per_environment, aggregate_environment_metrics(per_environment)


def evaluate_environments_detailed(
    model: TemporalDiffusionModel,
    environments: Sequence[PreparedEnvironment],
    batch_size: int,
    device: torch.device,
    max_batches: int,
    *,
    rerank_mode: str = "none",
    rerank_topk: int = 100,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, object]]:
    per_environment: list[dict[str, float]] = []
    merged = {
        "popularity": {
            name: RankingAccumulator((10, 50, 100)) for name in POPULARITY_GROUPS
        },
        "recency": {
            name: RankingAccumulator((10, 50, 100)) for name in RECENCY_GROUPS
        },
    }
    for environment in environments:
        accumulator, strata = evaluate_environment_accumulators(
            model,
            environment,
            batch_size,
            device,
            max_batches,
            stratified=True,
            rerank_mode=rerank_mode,
            rerank_topk=rerank_topk,
        )
        per_environment.append(accumulator.compute())
        for taxonomy, groups in strata.items():
            for name, group_accumulator in groups.items():
                merged[taxonomy][name].merge(group_accumulator)

    stratified_metrics: dict[str, object] = {}
    for taxonomy, groups in merged.items():
        stratified_metrics[taxonomy] = {}
        for name, accumulator in groups.items():
            values: dict[str, object] = {"count": accumulator.count}
            if accumulator.count:
                values.update(accumulator.compute())
            stratified_metrics[taxonomy][name] = values
    return (
        per_environment,
        aggregate_environment_metrics(per_environment),
        stratified_metrics,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.early_stopping_patience < 0 or args.minimum_epochs < 1:
        raise ValueError("early-stopping patience cannot be negative and minimum epochs must be positive")
    if min(args.preservation_weight, args.preservation_margin) < 0:
        raise ValueError("preservation weight and margin cannot be negative")
    if args.preservation_topk < 1:
        raise ValueError("preservation top-k must be positive")
    if args.freeze_backbone and args.initialize_checkpoint is None:
        raise ValueError("freezing the backbone requires --initialize-checkpoint")
    if (args.freeze_backbone or args.preservation_weight) and args.prior_mode != "temporal":
        raise ValueError("anchored residual training requires --prior-mode temporal")
    seed_everything(args.seed)
    device = select_device(args.device)
    (
        num_nodes,
        train_cpu,
        valid_cpu,
        train_records,
        valid_records,
        test_records,
    ) = build_prepared_protocol(args)
    train_environments = [environment.graph_to(device) for environment in train_cpu]
    valid_environments = [environment.graph_to(device) for environment in valid_cpu]

    model = TemporalDiffusionModel(
        num_nodes=num_nodes,
        dimension=args.dimension,
        rank=args.rank,
        context_dim=args.context_dim,
        environment_hidden_dim=args.environment_hidden_dim,
        dropout=args.dropout,
        mask_mode=args.mask_mode,
        prior_mode=args.prior_mode,
    ).to(device)
    initialization_checkpoint: dict[str, object] | None = None
    if args.initialize_checkpoint is not None:
        initialization_checkpoint = torch.load(
            args.initialize_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if initialization_checkpoint["num_nodes"] != num_nodes:
            raise ValueError("initialization checkpoint has a different node vocabulary")
        source_config = initialization_checkpoint["model_config"]
        requested_config = {
            "dimension": args.dimension,
            "rank": args.rank,
            "context_dim": args.context_dim,
            "environment_hidden_dim": args.environment_hidden_dim,
            "dropout": args.dropout,
            "mask_mode": args.mask_mode,
        }
        mismatched = {
            name: (source_config.get(name), value)
            for name, value in requested_config.items()
            if source_config.get(name) != value
        }
        if mismatched:
            raise ValueError(f"initialization model configuration mismatch: {mismatched}")
        base_state = {
            name: value
            for name, value in initialization_checkpoint["model_state_dict"].items()
            if not name.startswith("temporal_prior_gate.")
        }
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        if unexpected or any(
            not name.startswith("temporal_prior_gate.") for name in missing
        ):
            raise RuntimeError(
                f"incompatible initialization checkpoint: missing={missing}, "
                f"unexpected={unexpected}"
            )
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.temporal_prior_gate.parameters():
            parameter.requires_grad_(True)
    risk_objective: ERM | GroupDRO | VREx
    if args.objective == "groupdro":
        risk_objective = GroupDRO(
            len(train_environments), step_size=args.group_dro_step_size
        ).to(device)
    elif args.objective == "erm":
        risk_objective = ERM(len(train_environments)).to(device)
    else:
        risk_objective = VREx(
            len(train_environments), penalty_weight=args.vrex_weight
        ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loaders = [
        make_loader(environment, args.batch_size, shuffle=True, seed=args.seed + index)
        for index, environment in enumerate(train_environments)
    ]
    steps_per_epoch = args.steps_per_epoch or max(len(loader) for loader in loaders)
    iterators = [iter(loader) for loader in loaders]

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("run-name may contain only letters, digits, dot, dash, and underscore")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"{args.dataset}_{args.run_name}.pt"
    best_validation_map = float("-inf")
    epochs_without_improvement = 0
    history: list[EpochSummary] = []

    print(
        f"dataset={args.dataset} nodes={num_nodes} device={device} "
        f"mask={args.mask_mode} prior={args.prior_mode} objective={args.objective} "
        f"frozen_backbone={args.freeze_backbone} "
        f"trainable_parameters={sum(item.numel() for item in trainable_parameters)} "
        f"train_examples={[len(env.dataset) for env in train_environments]}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        risk_objective.train()
        epoch_loss = 0.0
        environment_prediction_sums = [0.0] * len(train_environments)
        environment_constraint_sums = [0.0] * len(train_environments)
        environment_preservation_sums = [0.0] * len(train_environments)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            environment_losses: list[torch.Tensor] = []
            prediction_risks: list[torch.Tensor] = []
            for index, (environment, loader) in enumerate(zip(train_environments, loaders)):
                try:
                    batch = next(iterators[index])
                except StopIteration:
                    iterators[index] = iter(loader)
                    batch = next(iterators[index])
                batch = batch_to_device(batch, device)
                output = forward_environment(model, environment, batch)
                breakdown = environment_loss(
                    output,
                    batch["target"],
                    environment.local_popularity,
                    environment.historical_popularity,
                    environment.recent_popularity,
                    environment.popularity_groups,
                    environment.recency_groups,
                    shortcut_weight=args.shortcut_weight,
                    independence_weight=args.independence_weight,
                    mask_balance_weight=args.mask_balance_weight,
                    popularity_balance_alpha=args.popularity_balance_alpha,
                    dormant_boost=args.dormant_boost,
                    constraint_weight=args.constraint_weight,
                    constraint_margin=args.constraint_margin,
                )
                preservation_penalty = output.logits.new_zeros(())
                if args.preservation_weight:
                    preservation_penalty = topk_subgroup_preservation_penalty(
                        output,
                        environment.popularity_groups,
                        environment.recency_groups,
                        topk=args.preservation_topk,
                        margin=args.preservation_margin,
                    )
                environment_losses.append(
                    breakdown.total
                    + args.preservation_weight * preservation_penalty
                )
                prediction_risks.append(breakdown.prediction)
                environment_prediction_sums[index] += float(breakdown.prediction.detach())
                environment_constraint_sums[index] += float(
                    breakdown.constraint_penalty.detach()
                )
                environment_preservation_sums[index] += float(
                    preservation_penalty.detach()
                )

            robust_loss = risk_objective(
                torch.stack(environment_losses),
                torch.stack(prediction_risks),
            )
            robust_loss.backward()
            clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_loss += float(robust_loss.detach())

        _, validation = evaluate_environments(
            model,
            valid_environments,
            args.batch_size,
            device,
            args.max_eval_batches,
        )
        summary = EpochSummary(
            epoch=epoch,
            training_loss=epoch_loss / steps_per_epoch,
            group_weights=risk_objective.weights.detach().cpu().tolist(),
            environment_prediction_loss=[
                value / steps_per_epoch for value in environment_prediction_sums
            ],
            environment_constraint_penalty=[
                value / steps_per_epoch for value in environment_constraint_sums
            ],
            environment_preservation_penalty=[
                value / steps_per_epoch for value in environment_preservation_sums
            ],
            validation=validation,
        )
        history.append(summary)
        print(
            f"epoch={epoch:03d} loss={summary.training_loss:.4f} "
            f"val_map@100={validation['map@100']:.4f} "
            f"val_worst_map@100={validation['worst_map@100']:.4f} "
            f"group_weights={[round(value, 3) for value in summary.group_weights]} "
            f"prediction_risk={[round(value, 3) for value in summary.environment_prediction_loss]} "
            f"constraint={[round(value, 3) for value in summary.environment_constraint_penalty]} "
            f"preservation={[round(value, 3) for value in summary.environment_preservation_penalty]}"
        )

        # The test environments are deliberately not referenced in this loop.
        if validation["map@100"] > best_validation_map + args.minimum_delta:
            best_validation_map = validation["map@100"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "num_nodes": num_nodes,
                    "model_config": {
                        "dimension": args.dimension,
                        "rank": args.rank,
                        "context_dim": args.context_dim,
                        "environment_hidden_dim": args.environment_hidden_dim,
                        "dropout": args.dropout,
                        "mask_mode": args.mask_mode,
                        "prior_mode": args.prior_mode,
                    },
                    "objective": args.objective,
                    "vrex_weight": args.vrex_weight,
                    "popularity_balance_alpha": args.popularity_balance_alpha,
                    "dormant_boost": args.dormant_boost,
                    "constraint_weight": args.constraint_weight,
                    "constraint_margin": args.constraint_margin,
                    "initialization_checkpoint": (
                        str(args.initialize_checkpoint.resolve())
                        if args.initialize_checkpoint
                        else None
                    ),
                    "freeze_backbone": args.freeze_backbone,
                    "preservation_weight": args.preservation_weight,
                    "preservation_topk": args.preservation_topk,
                    "preservation_margin": args.preservation_margin,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience
            and epoch >= args.minimum_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early_stop epoch={epoch} best_val_map@100={best_validation_map:.4f} "
                f"patience={args.early_stopping_patience}"
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        validation_by_environment,
        restored_validation,
        validation_stratified,
    ) = evaluate_environments_detailed(
        model,
        valid_environments,
        args.batch_size,
        device,
        args.max_eval_batches,
    )
    result: dict[str, object] = {
        "dataset": args.dataset,
        "num_nodes": num_nodes,
        "checkpoint": str(checkpoint_path),
        "mask_mode": args.mask_mode,
        "prior_mode": args.prior_mode,
        "objective": args.objective,
        "vrex_weight": args.vrex_weight,
        "method_label": args.method_label or f"{args.mask_mode}_{args.objective}",
        "popularity_balance_alpha": args.popularity_balance_alpha,
        "dormant_boost": args.dormant_boost,
        "constraint_weight": args.constraint_weight,
        "constraint_margin": args.constraint_margin,
        "initialization_checkpoint": (
            str(args.initialize_checkpoint.resolve())
            if args.initialize_checkpoint
            else None
        ),
        "freeze_backbone": args.freeze_backbone,
        "preservation_weight": args.preservation_weight,
        "preservation_topk": args.preservation_topk,
        "preservation_margin": args.preservation_margin,
        "seed": args.seed,
        "selected_epoch": checkpoint["epoch"],
        "trained_epochs": len(history),
        "selected_validation": checkpoint["validation"],
        "restored_validation": restored_validation,
        "validation_by_environment": validation_by_environment,
        "validation_stratified": validation_stratified,
        "history": [asdict(item) for item in history],
        "protocol": {
            "split": "chronological_70_10_20_timestamp_ties_preserved",
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "train_environments": args.train_environments,
            "valid_environments": args.valid_environments,
            "test_environments": args.test_environments,
            "sample_hop": args.sample_hop,
            "max_eval_batches": args.max_eval_batches,
            "test_materialized": not args.skip_test,
        },
    }
    if args.skip_test:
        print(
            f"validation_only checkpoint_epoch={checkpoint['epoch']} "
            f"map@100={restored_validation['map@100']:.4f} "
            f"worst_map@100={restored_validation['worst_map@100']:.4f}"
        )
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    test_environments = [
        environment.graph_to(device)
        for environment in prepare_final_test(
            args,
            num_nodes,
            train_records,
            valid_records,
            test_records,
        )
    ]
    per_test_environment, test_metrics, test_stratified = evaluate_environments_detailed(
        model,
        test_environments,
        args.batch_size,
        device,
        args.max_eval_batches,
    )
    print(
        f"final_test checkpoint_epoch={checkpoint['epoch']} "
        f"map@100={test_metrics['map@100']:.4f} "
        f"worst_map@100={test_metrics['worst_map@100']:.4f}"
    )
    result.update(
        test=test_metrics,
        test_by_environment=per_test_environment,
        test_stratified=test_stratified,
    )
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
