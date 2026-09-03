"""Retrain the unchanged CIKM DeDiff model with held-out-safe temporal splits.

This runner intentionally imports the original top-level ``model.py`` and
``module.py`` without editing them. It replaces their test-coupled runner and
legacy split files, constructs graph/frequency inputs from preceding records,
and performs validation-only checkpoint selection. The quadratic dense graph
interface of the original model is retained and reported as a scalability
limitation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .data import build_rolling_snapshots, chronological_split, load_cascades, make_temporal_environments
from .metrics import RankingAccumulator, aggregate_environment_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These imports resolve to the unchanged CIKM source in D:\DeDiff.
from graph import build_two_graphs  # type: ignore  # noqa: E402
from model import DeDiff, loss_function  # type: ignore  # noqa: E402


DIMENSIONS = {"christian": 64, "android": 128, "douban": 64, "twitter": 128}
HEADS = {"christian": 6, "android": 6, "douban": 8, "twitter": 10}
CUTOFFS = (10, 50, 100)


class TemporalDeDiffDataset(Dataset[dict[str, torch.Tensor]]):
    """Build the tensor dictionary consumed by unchanged DeDiff.forward."""

    def __init__(
        self,
        records: Sequence,
        distance_matrix: np.ndarray,
        *,
        max_length: int,
        eos_id: int,
    ) -> None:
        self.records = tuple(records)
        self.distance_matrix = distance_matrix
        self.max_length = max_length
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        users = list(record.cascade[: self.max_length + 1])
        timestamps = list(record.timestamp[: self.max_length + 1])
        if len(users) < 2:
            raise ValueError("next-user training requires at least two activations")
        input_users = users[:-1]
        distance = self.distance_matrix[np.ix_(input_users, input_users)]

        cascade = torch.zeros(self.max_length + 1, dtype=torch.long)
        timestamp = torch.zeros(self.max_length + 1, dtype=torch.float32)
        cascade[: len(users)] = torch.tensor(users, dtype=torch.long)
        timestamp[: len(timestamps)] = torch.tensor(timestamps, dtype=torch.float32)
        padded_distance = torch.full(
            (self.max_length, self.max_length),
            self.eos_id,
            dtype=torch.float32,
        )
        padded_distance[: len(input_users), : len(input_users)] = torch.as_tensor(
            distance,
            dtype=torch.float32,
        )
        # Neighbor/relation tensors are part of the legacy data contract but
        # are not read by DeDiff.forward.
        return {
            "cascade": cascade,
            "timestamp": timestamp,
            "neighbor": torch.zeros(self.max_length, 1, dtype=torch.long),
            "relation": torch.zeros(self.max_length, 1, dtype=torch.long),
            "dis_matrix": padded_distance,
        }


@dataclass
class DeDiffEnvironment:
    name: str
    dataset: TemporalDeDiffDataset
    info: dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DIMENSIONS), default="christian")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)
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


def model_arguments(args: argparse.Namespace, user_num: int) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        user_num=user_num,
        dim=DIMENSIONS[args.dataset],
        n_heads=HEADS[args.dataset],
        max_len=args.max_prefix_length,
        time_intervals=20,
        dropout=0.3,
        gcn_layer=1,
        sample_hop=2,
        device=torch.device("cuda"),
    )


def build_info(
    model_args: SimpleNamespace,
    history_records: Sequence,
    popularity: np.ndarray,
) -> dict[str, torch.Tensor]:
    edges: set[tuple[int, int]] = set()
    for record in history_records:
        cascade = record.cascade
        for destination in range(1, len(cascade)):
            start = max(0, destination - model_args.sample_hop)
            for source in range(start, destination):
                edges.add((cascade[source], cascade[destination]))
                edges.add((cascade[destination], cascade[source]))
    graph_stack = build_two_graphs(model_args, list(edges)).to(model_args.device)
    frequency = torch.from_numpy(popularity.astype(np.float32, copy=False))
    denominator = max(float(frequency.sum()) * 2.0, 1.0)
    frequency = (frequency / denominator).to(model_args.device)
    if frequency.numel() < model_args.user_num:
        frequency = torch.nn.functional.pad(
            frequency,
            (0, model_args.user_num - frequency.numel()),
        )
    return {
        # Preserve the exact channel order used by original graph.get_info.
        "A_interaction": graph_stack[:, :, 0],
        "A_social": graph_stack[:, :, 1],
        "frequency": frequency,
    }


def prepare_protocol(
    args: argparse.Namespace,
) -> tuple[SimpleNamespace, list[DeDiffEnvironment], list[DeDiffEnvironment], int, int]:
    records = load_cascades(args.dataset_root / args.dataset)
    split = chronological_split(records)
    max_user_id = max(node for record in records for node in record.cascade)
    user_num = max_user_id + 2  # PAD=0 and EOS=max_user_id+1.
    model_args = model_arguments(args, user_num)
    model_args.social_graph_path = str(args.dataset_root / args.dataset / "social_graph.npz")
    distance = np.load(args.dataset_root / args.dataset / "distance.npy", allow_pickle=True)
    train_groups = make_temporal_environments(
        split.train,
        args.train_environments,
        prefix="train",
    )
    valid_groups = make_temporal_environments(
        split.valid,
        args.valid_environments,
        prefix="valid",
    )
    train_snapshots = build_rolling_snapshots(
        train_groups,
        max_user_id + 1,
        sample_hop=2,
    )
    valid_snapshots = build_rolling_snapshots(
        valid_groups,
        max_user_id + 1,
        warm_start_records=split.train,
        warm_start_recent_records=train_groups[-1].records,
        sample_hop=2,
    )

    def convert(snapshots) -> list[DeDiffEnvironment]:
        converted = []
        for snapshot in snapshots:
            converted.append(
                DeDiffEnvironment(
                    name=snapshot.environment.name,
                    dataset=TemporalDeDiffDataset(
                        snapshot.environment.records,
                        distance,
                        max_length=args.max_prefix_length,
                        eos_id=user_num - 1,
                    ),
                    info=build_info(
                        model_args,
                        # RollingSnapshot intentionally exposes only aggregate
                        # tensors; reconstruct the strictly preceding records.
                        tuple(
                            record
                            for record in (*split.train, *split.valid)
                            if record.start_time < snapshot.environment.start_time
                        )[: snapshot.history_size],
                        snapshot.popularity,
                    ),
                )
            )
        return converted

    # Test records remain immutable CascadeRecord objects and are never passed
    # to Dataset, graph, tensor, or model construction.
    return model_args, convert(train_snapshots), convert(valid_snapshots), len(split.test), max_user_id + 1


def make_loaders(
    environments: Sequence[DeDiffEnvironment],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[DataLoader]:
    return [
        DataLoader(
            environment.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed + index),
            num_workers=0,
            pin_memory=True,
        )
        for index, environment in enumerate(environments)
    ]


def train_epoch(
    model: DeDiff,
    model_args: SimpleNamespace,
    environments: Sequence[DeDiffEnvironment],
    loaders: Sequence[DataLoader],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    gradient_clip: float,
) -> float:
    model.train()
    iterators = [iter(loader) for loader in loaders]
    total = 0.0
    for step in range(steps):
        environment_index = step % len(loaders)
        try:
            batch = next(iterators[environment_index])
        except StopIteration:
            iterators[environment_index] = iter(loaders[environment_index])
            batch = next(iterators[environment_index])
        prediction, label, state = model(
            model_args,
            batch,
            environments[environment_index].info,
        )
        loss = loss_function(prediction, label, state)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite DeDiff loss: {float(loss.detach())}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total += float(loss.detach())
    return total / steps


@torch.no_grad()
def evaluate(
    model: DeDiff,
    model_args: SimpleNamespace,
    environments: Sequence[DeDiffEnvironment],
    loaders: Sequence[DataLoader],
    *,
    max_batches: int,
    num_nodes: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    metrics = []
    for environment, loader in zip(environments, loaders):
        accumulator = RankingAccumulator(CUTOFFS)
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            prediction, label, _ = model(model_args, batch, environment.info)
            valid = label.ne(0) & label.ne(model_args.user_num - 1)
            target = label[valid]
            # Candidate index equals the original user id. Candidate zero is
            # retained for exact alignment with the common max-id vocabulary,
            # but is impossible because its score is -inf.
            real_scores = torch.cat(
                [
                    prediction.new_full((prediction.shape[0], 1), float("-inf")),
                    prediction[:, 1:-1],
                ],
                dim=1,
            )
            if real_scores.shape[1] != num_nodes:
                raise RuntimeError("DeDiff candidate vocabulary is misaligned")
            accumulator.update(real_scores[valid], target)
        metrics.append(accumulator.compute())
    return aggregate_environment_metrics(metrics), metrics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the unchanged CIKM implementation requires CUDA")
    seed_everything(args.seed)
    model_args, train_environments, valid_environments, test_count, num_nodes = prepare_protocol(args)
    train_loaders = make_loaders(
        train_environments,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    valid_loaders = make_loaders(
        valid_environments,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    model = DeDiff(model_args).to(model_args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    selected_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            model_args,
            train_environments,
            train_loaders,
            optimizer,
            steps=args.steps_per_epoch,
            gradient_clip=args.gradient_clip,
        )
        validation, by_environment = evaluate(
            model,
            model_args,
            valid_environments,
            valid_loaders,
            max_batches=args.max_eval_batches,
            num_nodes=num_nodes,
        )
        score = validation["map@100"]
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss,
                "validation": validation,
                "validation_by_environment": by_environment,
            }
        )
        print(f"epoch={epoch:03d} loss={loss:.6f} valid_map@100={score:.6f}", flush=True)
        if score > best:
            best = score
            selected_epoch = epoch
            stale = 0
            torch.save(
                {"model_state": model.state_dict(), "selected_epoch": epoch},
                args.checkpoint,
            )
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    checkpoint = torch.load(args.checkpoint, map_location=model_args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation, by_environment = evaluate(
        model,
        model_args,
        valid_environments,
        valid_loaders,
        max_batches=args.max_eval_batches,
        num_nodes=num_nodes,
    )
    result = {
        "status": "validation_only_corrected_dediff",
        "dataset": args.dataset,
        "seed": args.seed,
        "selected_epoch": selected_epoch,
        "checkpoint": str(args.checkpoint.resolve()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "validation": validation,
        "validation_by_environment": by_environment,
        "history": history,
        "counts": {"test_retained_not_materialized": test_count},
        "protocol": {
            "source_model": "unchanged CIKM DeDiff model.py/module.py",
            "chronological_split": [0.7, 0.1, 0.2],
            "timestamp_ties_preserved": True,
            "train_environments": args.train_environments,
            "validation_environments": args.valid_environments,
            "rolling_past_only_graph_inputs": True,
            "selection_metric": "mean validation MAP@100",
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "postfreeze_descriptive": True,
        },
        "model_arguments": vars(args),
    }
    # Convert Path values explicitly for stable JSON provenance.
    result["model_arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result["model_arguments"].items()
    }
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_epoch": selected_epoch,
        "parameter_count": result["parameter_count"],
        "validation": validation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
