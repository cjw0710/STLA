"""Attach the temporal logit adapter to a frozen corrected task baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .baselines.buzzbloom_temporal import (
    TemporalBuzzLoader,
    _buzz_cascade_dataset,
    apply_semantics_preserving_patches,
    load_model_class,
    make_collate_fn,
    select_device,
)
from .data import build_rolling_snapshots, make_temporal_environments
from .metrics import RankingAccumulator, aggregate_environment_metrics, protected_union_scores
from .models import TemporalLogitAdapter
from .training import prepare_environment


ROOT = Path(__file__).resolve().parent
CUTOFFS = (10, 50, 100)
PATHS = ("anchor", "adaptive", "hierarchical_union")


@dataclass
class AdapterEnvironment:
    name: str
    dataset: Any
    environment_features: torch.Tensor
    historical_popularity: torch.Tensor
    recent_popularity: torch.Tensor
    popularity_groups: torch.Tensor
    recency_groups: torch.Tensor

    def context_to(self, device: torch.device) -> "AdapterEnvironment":
        return AdapterEnvironment(
            name=self.name,
            dataset=self.dataset,
            environment_features=self.environment_features.to(device),
            historical_popularity=self.historical_popularity.to(device),
            recent_popularity=self.recent_popularity.to(device),
            popularity_groups=self.popularity_groups.to(device),
            recency_groups=self.recency_groups.to(device),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--environment-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument(
        "--ablation",
        choices=TemporalLogitAdapter.ablations,
        default="full",
        help="validation-only adapter component ablation; default preserves prior behavior",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _record_key(record) -> tuple[str, int, float]:
    return record.source_split, record.source_index, record.start_time


def build_adapter_environments(
    loader: TemporalBuzzLoader,
    *,
    train_count: int,
    valid_count: int,
    max_prefix_length: int,
) -> tuple[list[AdapterEnvironment], list[AdapterEnvironment]]:
    train_groups = make_temporal_environments(loader.split.train, train_count, prefix="train")
    valid_groups = make_temporal_environments(loader.split.valid, valid_count, prefix="valid")
    train_snapshots = build_rolling_snapshots(train_groups, loader.num_nodes, sample_hop=2)
    valid_snapshots = build_rolling_snapshots(
        valid_groups,
        loader.num_nodes,
        warm_start_records=loader.split.train,
        warm_start_recent_records=train_groups[-1].records,
        sample_hop=2,
    )
    train_indices = {
        _record_key(record): index
        for index, record in enumerate(loader.split.train, start=1)
    }
    CascadeDataset = _buzz_cascade_dataset()

    def convert(snapshots, *, validation: bool) -> list[AdapterEnvironment]:
        converted: list[AdapterEnvironment] = []
        for snapshot in snapshots:
            records = snapshot.environment.records
            cascades = [[node + 2 for node in record.cascade] for record in records]
            timestamps = [list(record.timestamp) for record in records]
            indices = (
                [0] * len(records)
                if validation
                else [train_indices[_record_key(record)] for record in records]
            )
            prepared = prepare_environment(snapshot, loader.num_nodes, max_prefix_length)
            converted.append(
                AdapterEnvironment(
                    name=snapshot.environment.name,
                    dataset=CascadeDataset(cascades, timestamps, indices),
                    environment_features=prepared.environment_features,
                    historical_popularity=prepared.historical_popularity,
                    recent_popularity=prepared.recent_popularity,
                    popularity_groups=prepared.popularity_groups,
                    recency_groups=prepared.recency_groups,
                )
            )
        return converted

    return convert(train_snapshots, validation=False), convert(valid_snapshots, validation=True)


def make_loaders(
    environments: Sequence[AdapterEnvironment],
    *,
    batch_size: int,
    max_prefix_length: int,
    shuffle: bool,
    seed: int,
) -> list[DataLoader]:
    collate = make_collate_fn(max_prefix_length)
    return [
        DataLoader(
            environment.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed + index),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
        for index, environment in enumerate(environments)
    ]


def load_frozen_anchor(
    baseline: dict[str, Any],
    loader: TemporalBuzzLoader,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    protocol = baseline.get("protocol", {})
    if any(protocol.get(field) is not False for field in (
        "test_materialized", "test_evaluated", "test_used_for_selection"
    )):
        raise ValueError("baseline result did not pass the validation-only protocol")
    arguments = dict(baseline["model_arguments"])
    arguments["device"] = device
    model_name = baseline["model_name"]
    Model = load_model_class(model_name)
    anchor = Model(SimpleNamespace(**arguments), loader).to(device)
    patches = apply_semantics_preserving_patches(anchor, model_name)
    checkpoint = torch.load(Path(baseline["checkpoint"]), map_location=device, weights_only=False)
    anchor.load_state_dict(checkpoint["model_state"])
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    return anchor, patches


def _anchor_logits(
    anchor: torch.nn.Module,
    sequence: torch.Tensor,
    timestamp: torch.Tensor,
    cascade_index: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return anchor(sequence, timestamp, cascade_index)[:, 2:]


def train_epoch(
    anchor: torch.nn.Module,
    adapter: TemporalLogitAdapter,
    environments: Sequence[AdapterEnvironment],
    loaders: Sequence[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    steps: int,
    gradient_clip: float,
    ablation: str = "full",
) -> float:
    adapter.train()
    iterators = [iter(loader) for loader in loaders]
    total_loss = 0.0
    for step in range(steps):
        environment_index = step % len(loaders)
        try:
            batch = next(iterators[environment_index])
        except StopIteration:
            iterators[environment_index] = iter(loaders[environment_index])
            batch = next(iterators[environment_index])
        sequence, timestamp, cascade_index = (
            tensor.to(device, non_blocking=True) for tensor in batch
        )
        environment = environments[environment_index]
        anchor_logits = _anchor_logits(anchor, sequence, timestamp, cascade_index)
        adaptive_logits, _ = adapter(
            anchor_logits,
            sequence,
            timestamp,
            environment.environment_features,
            environment.historical_popularity,
            environment.recent_popularity,
            ablation=ablation,
        )
        gold = sequence[:, 1:].reshape(-1)
        valid = gold.ne(0)
        loss = torch.nn.functional.cross_entropy(adaptive_logits[valid], gold[valid] - 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(adapter.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / steps


@torch.no_grad()
def evaluate(
    anchor: torch.nn.Module,
    adapter: TemporalLogitAdapter,
    environments: Sequence[AdapterEnvironment],
    loaders: Sequence[DataLoader],
    device: torch.device,
    *,
    max_batches: int,
    ablation: str = "full",
) -> dict[str, Any]:
    anchor.eval()
    adapter.eval()
    by_path = {path: [] for path in PATHS}
    guarantee = {
        str(cutoff): {"protected_anchor_hits": 0, "violations": 0}
        for cutoff in CUTOFFS
    }
    for environment, loader in zip(environments, loaders):
        accumulators = {path: RankingAccumulator(CUTOFFS) for path in PATHS}
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            sequence, timestamp, cascade_index = (
                tensor.to(device, non_blocking=True) for tensor in batch
            )
            anchor_all = _anchor_logits(anchor, sequence, timestamp, cascade_index)
            adaptive_all, _ = adapter(
                anchor_all,
                sequence,
                timestamp,
                environment.environment_features,
                environment.historical_popularity,
                environment.recent_popularity,
                ablation=ablation,
            )
            gold = sequence[:, 1:].reshape(-1)
            valid = gold.ne(0)
            target = gold[valid] - 2
            anchor_logits = anchor_all[valid]
            adaptive_logits = adaptive_all[valid]
            fused_logits = protected_union_scores(
                adaptive_logits,
                anchor_logits,
                environment.popularity_groups,
                environment.recency_groups,
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
                (environment.popularity_groups[target] != 0)
                | (environment.recency_groups[target] != 0)
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
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    seed = int(baseline["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    max_prefix_length = int(baseline["protocol"]["max_prefix_length"])
    loader = TemporalBuzzLoader(
        baseline["dataset"],
        args.dataset_root,
        max_prefix_length=max_prefix_length,
        valid_environments=args.valid_environments,
    )
    train_environments, valid_environments = build_adapter_environments(
        loader,
        train_count=args.train_environments,
        valid_count=args.valid_environments,
        max_prefix_length=max_prefix_length,
    )
    train_environments = [environment.context_to(device) for environment in train_environments]
    valid_environments = [environment.context_to(device) for environment in valid_environments]
    train_loaders = make_loaders(
        train_environments,
        batch_size=args.batch_size,
        max_prefix_length=max_prefix_length,
        shuffle=True,
        seed=seed,
    )
    valid_loaders = make_loaders(
        valid_environments,
        batch_size=args.batch_size,
        max_prefix_length=max_prefix_length,
        shuffle=False,
        seed=seed,
    )
    anchor, patches = load_frozen_anchor(baseline, loader, device)
    adapter = TemporalLogitAdapter(
        loader.num_nodes,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        environment_hidden_dim=args.environment_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)

    initial = evaluate(
        anchor,
        adapter,
        valid_environments,
        valid_loaders,
        device,
        max_batches=args.max_eval_batches,
        ablation=args.ablation,
    )
    best_score = initial["paths"]["hierarchical_union"]["metrics"]["map@100"]
    selected_epoch = 0
    torch.save(
        {
            "adapter_state": adapter.state_dict(),
            "selected_epoch": 0,
            "adapter_ablation": args.ablation,
        },
        args.checkpoint,
    )
    history: list[dict[str, Any]] = [{"epoch": 0, "training_loss": None, "evaluation": initial}]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            anchor,
            adapter,
            train_environments,
            train_loaders,
            optimizer,
            device,
            steps=args.steps_per_epoch,
            gradient_clip=args.gradient_clip,
            ablation=args.ablation,
        )
        evaluation = evaluate(
            anchor,
            adapter,
            valid_environments,
            valid_loaders,
            device,
            max_batches=args.max_eval_batches,
            ablation=args.ablation,
        )
        score = evaluation["paths"]["hierarchical_union"]["metrics"]["map@100"]
        history.append({"epoch": epoch, "training_loss": loss, "evaluation": evaluation})
        print(f"epoch={epoch:03d} loss={loss:.6f} fused_map@100={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            stale = 0
            torch.save(
                {
                    "adapter_state": adapter.state_dict(),
                    "selected_epoch": epoch,
                    "adapter_ablation": args.ablation,
                },
                args.checkpoint,
            )
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    selected = torch.load(args.checkpoint, map_location=device, weights_only=False)
    adapter.load_state_dict(selected["adapter_state"])
    final = evaluate(
        anchor,
        adapter,
        valid_environments,
        valid_loaders,
        device,
        max_batches=args.max_eval_batches,
        ablation=args.ablation,
    )
    result = {
        "status": "validation_only_strong_backbone_gate",
        "dataset": baseline["dataset"],
        "seed": seed,
        "anchor_model": baseline["model_name"],
        "adapter_ablation": args.ablation,
        "anchor_result": str(args.baseline_result.resolve()),
        "anchor_checkpoint": baseline["checkpoint"],
        "selected_epoch": selected_epoch,
        "adapter_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "evaluation": final,
        "history": history,
        "semantics_preserving_anchor_patches": patches,
        "protocol": {
            "chronological_split": [0.7, 0.1, 0.2],
            "train_environments": args.train_environments,
            "validation_environments": args.valid_environments,
            "past_only_adapter_features": True,
            "anchor_frozen": True,
            "selection_metric": "hierarchical union mean validation MAP@100",
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "selection_changes_permitted": False,
            "postfreeze_descriptive": True,
            "post_confirmation_validation_analysis": args.ablation != "full",
            "confirmatory_test_reused": False,
        },
    }
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_epoch": selected_epoch,
        "adapter_parameter_count": result["adapter_parameter_count"],
        "metrics": final["paths"]["hierarchical_union"]["metrics"],
        "guarantee": final["guarantee"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
