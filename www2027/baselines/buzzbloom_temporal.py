"""Train BuzzBloom-integrated baselines under the DeDiff temporal protocol.

This adapter deliberately does not call BuzzBloom's runner or data splitter.
It reuses only the model implementations and their graph builders while
enforcing DeDiff's tie-preserving 70/10/20 chronological split, train-only
graph construction, two validation environments, and validation-only model
selection.  The held-out test partition is never materialized as a Dataset or
DataLoader by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import pickle
import random
import sys
import types
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..data import (
    CascadeRecord,
    TemporalSplit,
    chronological_split,
    load_cascades,
    make_temporal_environments,
)
from ..metrics import RankingAccumulator, aggregate_environment_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
WWW_ROOT = Path(__file__).resolve().parents[1]
BUZZBLOOM_ROOT = WWW_ROOT / "third_party" / "buzzbloom"
SUPPORTED_MODELS = ("DyHGCN", "MSHGAT", "DisenIDP")
PAD = 0
EOS = 1
PAD_TIME = -1.0


def _activate_buzzbloom() -> None:
    """Make BuzzBloom's absolute imports resolve without editing its source."""

    if not (BUZZBLOOM_ROOT / "models").is_dir():
        raise FileNotFoundError(
            f"BuzzBloom source is missing at {BUZZBLOOM_ROOT}. "
            "See www2027/third_party/README.md."
        )
    source = str(BUZZBLOOM_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def load_model_class(model_name: str) -> type[torch.nn.Module]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}")
    _activate_buzzbloom()
    module = importlib.import_module(f"models.{model_name}")
    return getattr(module, model_name)


def _buzz_cascade_dataset() -> type:
    _activate_buzzbloom()
    module = importlib.import_module("helpers.BaseLoader")
    return module.CascadeDataset


def make_collate_fn(max_prefix_length: int) -> Callable[[list[tuple]], tuple[torch.Tensor, ...]]:
    """Return BuzzBloom-compatible padding capped at prefix length plus target."""

    if max_prefix_length < 1:
        raise ValueError("max_prefix_length must be positive")

    def collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not batch:
            raise ValueError("cannot collate an empty batch")
        cascades, timestamps, indices = zip(*batch)
        max_len = min(max(len(sequence) for sequence in cascades), max_prefix_length + 1)
        if max_len < 2:
            raise ValueError("every training/evaluation cascade must contain at least two users")

        padded_cascades: list[list[int]] = []
        padded_timestamps: list[list[float]] = []
        for cascade, timestamp in zip(cascades, timestamps):
            users = list(cascade[:max_len])
            times = list(timestamp[:max_len])
            users.extend([PAD] * (max_len - len(users)))
            times.extend([PAD_TIME] * (max_len - len(times)))
            padded_cascades.append(users)
            padded_timestamps.append(times)

        return (
            torch.tensor(padded_cascades, dtype=torch.long),
            torch.tensor(padded_timestamps, dtype=torch.float32),
            torch.tensor(indices, dtype=torch.long),
        )

    return collate


def _shift_records(
    records: Sequence[CascadeRecord],
    *,
    first_index: int,
    validation_indices_are_dummy: bool,
) -> tuple[list[list[int]], list[list[float]], list[int]]:
    cascades = [[node + 2 for node in record.cascade] for record in records]
    timestamps = [list(record.timestamp) for record in records]
    if validation_indices_are_dummy:
        # MSHGAT otherwise indexes a train-built cascade embedding with the
        # validation cascade id, which either leaks the full cascade or is OOB.
        indices = [0] * len(records)
    else:
        # BuzzBloom hypergraph cascade column zero is reserved as a dummy.
        indices = list(range(first_index, first_index + len(records)))
    return cascades, timestamps, indices


def _infer_num_nodes(records: Sequence[CascadeRecord]) -> int:
    return max(node for record in records for node in record.cascade) + 1


class TemporalBuzzLoader:
    """The small loader interface expected by the three BuzzBloom models."""

    def __init__(
        self,
        dataset: str,
        dataset_root: Path,
        *,
        max_prefix_length: int,
        valid_environments: int = 2,
        mapping_root: Path | None = None,
    ) -> None:
        if valid_environments < 1:
            raise ValueError("valid_environments must be positive")
        dataset_dir = dataset_root / dataset
        split_manifest_path = dataset_dir / "split_manifest.json"
        if split_manifest_path.is_file():
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            split_protocol = split_manifest.get("split", {})
            if split_protocol.get("strict_train_valid_only_during_selection") is not True:
                raise ValueError("prepartitioned dataset manifest does not lock test access")
            train_records = load_cascades(dataset_dir, split_names=("train",))
            valid_records = load_cascades(dataset_dir, split_names=("valid",))
            records = (*train_records, *valid_records)
            split = TemporalSplit(train=train_records, valid=valid_records, test=tuple())
            declared_counts = split_manifest.get("counts", {})
            self.test_record_count = int(declared_counts["test"])
            self.total_record_count = int(declared_counts["all"])
            declared_nodes = int(declared_counts["nodes"])
            if len(train_records) != int(declared_counts["train"]):
                raise ValueError("training count differs from the split manifest")
            if len(valid_records) != int(declared_counts["valid"]):
                raise ValueError("validation count differs from the split manifest")
            self.strict_prepartitioned = True
        else:
            records = load_cascades(dataset_dir)
            split = chronological_split(records, ratios=(0.7, 0.1, 0.2))
            self.test_record_count = len(split.test)
            self.total_record_count = len(records)
            declared_nodes = _infer_num_nodes(records)
            self.strict_prepartitioned = False

        self.dataset = dataset
        self.records = records
        self.split = split
        self.num_nodes = declared_nodes
        self.user_num = self.num_nodes + 2
        self.max_prefix_length = max_prefix_length
        self.test_materialized = False

        train_cascades, train_timestamps, train_indices = _shift_records(
            split.train,
            first_index=1,
            validation_indices_are_dummy=False,
        )
        CascadeDataset = _buzz_cascade_dataset()
        self.train_set = CascadeDataset(train_cascades, train_timestamps, train_indices)

        validation = make_temporal_environments(
            split.valid,
            valid_environments,
            prefix="valid",
        )
        self.valid_environment_names = [environment.name for environment in validation]
        self.valid_sets = []
        for environment in validation:
            cascades, timestamps, indices = _shift_records(
                environment.records,
                first_index=0,
                validation_indices_are_dummy=True,
            )
            self.valid_sets.append(CascadeDataset(cascades, timestamps, indices))

        # MSHGAT expects an EOS-terminated graph-only view and drops the final
        # item.  The prediction Dataset above never contains EOS, so EOS cannot
        # become a target or metric candidate.
        self.cascades = [cascade + [EOS] for cascade in train_cascades]
        self.timestamps = [
            timestamp + [timestamp[-1]] for timestamp in train_timestamps
        ]
        self.all_cascades = self.cascades
        self.train_cas_user_dict = {
            index: list(cascade)
            for index, cascade in zip(train_indices, train_cascades)
        }
        self.cas_num = max(len(train_cascades) + 1, max_prefix_length + 1)

        artifact_root = mapping_root or WWW_ROOT / "artifacts" / "baseline_mappings"
        mapping_dir = artifact_root / dataset
        mapping_dir.mkdir(parents=True, exist_ok=True)
        self.u2idx_dict = str(mapping_dir / "u2idx.pickle")
        self.idx2u_dict = str(mapping_dir / "idx2u.pickle")
        self.net_data = str(dataset_dir / "graph.txt")
        self._write_identity_mappings()

    def _write_identity_mappings(self) -> None:
        u2idx: dict[str, int] = {"<blank>": PAD, "</s>": EOS}
        u2idx.update({str(node): node + 2 for node in range(self.num_nodes)})
        idx2u: list[str] = ["<blank>", "</s>"] + [
            str(node) for node in range(self.num_nodes)
        ]
        with Path(self.u2idx_dict).open("wb") as stream:
            pickle.dump(u2idx, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with Path(self.idx2u_dict).open("wb") as stream:
            pickle.dump(idx2u, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "all": self.total_record_count,
            "train": len(self.split.train),
            "valid": len(self.split.valid),
            "test_retained_not_materialized": self.test_record_count,
        }


@dataclass
class EpochResult:
    epoch: int
    training_loss: float
    validation: dict[str, float]
    validation_by_environment: dict[str, dict[str, float]]


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


def _iter_steps(loader: DataLoader, steps: int) -> Iterable[tuple[torch.Tensor, ...]]:
    if steps < 1:
        yield from loader
        return
    iterator = iter(loader)
    for _ in range(steps):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def vectorized_mshgat_previous_user_mask(
    self: torch.nn.Module,
    sequence: torch.Tensor,
    user_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Exact vectorization of BuzzBloom MSHGAT's nested masking loops."""

    if sequence.ndim != 2:
        raise ValueError("sequence must have shape [batch, length]")
    sequence = sequence.to(device)
    batch_size, length = sequence.shape
    # For prediction position t, retain sequence ids from source positions
    # 0..t and replace later ids with PAD. Scattering -1000 therefore masks
    # exactly the previous/current users and always masks PAD, matching the
    # upstream implementation without its two nested Python loops.
    candidate_indices = sequence.unsqueeze(1).expand(batch_size, length, length)
    positions = torch.arange(length, device=device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    masked_indices = torch.where(
        causal.unsqueeze(0),
        candidate_indices,
        torch.zeros_like(candidate_indices),
    )
    result = torch.zeros(batch_size, length, user_size, device=device)
    result.scatter_(2, masked_indices.long(), -1000.0)
    result[:, :, PAD] = -1000.0
    result.requires_grad_(False)
    return result


def apply_semantics_preserving_patches(
    model: torch.nn.Module,
    model_name: str,
) -> list[str]:
    """Apply adapter-local performance fixes that leave model outputs exact."""

    applied: list[str] = []
    if model_name == "MSHGAT":
        model.get_previous_user_mask = types.MethodType(  # type: ignore[method-assign]
            vectorized_mshgat_previous_user_mask,
            model,
        )
        applied.append("vectorized_previous_user_mask_exact")
    return applied


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    device: torch.device,
    *,
    steps_per_epoch: int,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for sequence, timestamp, cascade_index in _iter_steps(loader, steps_per_epoch):
        sequence = sequence.to(device, non_blocking=True)
        timestamp = timestamp.to(device, non_blocking=True)
        cascade_index = cascade_index.to(device, non_blocking=True)
        gold = sequence[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.get_performance(
            sequence,
            timestamp,
            cascade_index,
            loss_function,
            gold,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss: {float(loss.detach())}")
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return total_loss / batches


@torch.no_grad()
def evaluate_environment(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_eval_batches: int,
) -> dict[str, float]:
    model.eval()
    accumulator = RankingAccumulator()
    for batch_index, (sequence, timestamp, cascade_index) in enumerate(loader):
        if max_eval_batches and batch_index >= max_eval_batches:
            break
        sequence = sequence.to(device, non_blocking=True)
        timestamp = timestamp.to(device, non_blocking=True)
        cascade_index = cascade_index.to(device, non_blocking=True)
        gold = sequence[:, 1:].reshape(-1)
        prediction = model(sequence, timestamp, cascade_index)
        valid = gold.ne(PAD)
        # Remove special-token columns. Real users return to DeDiff's original
        # zero-based candidate ids before exact rank computation.
        accumulator.update(prediction[valid, 2:], gold[valid] - 2)
    return accumulator.compute()


def evaluate_validation(
    model: torch.nn.Module,
    loaders: Sequence[DataLoader],
    names: Sequence[str],
    device: torch.device,
    *,
    max_eval_batches: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    by_environment = {
        name: evaluate_environment(
            model,
            loader,
            device,
            max_eval_batches=max_eval_batches,
        )
        for name, loader in zip(names, loaders)
    }
    aggregate = aggregate_environment_metrics(by_environment.values())
    return aggregate, by_environment


def _base_parser() -> argparse.ArgumentParser:
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--model-name", choices=SUPPORTED_MODELS, default="DyHGCN")
    known, _ = initial.parse_known_args()
    Model = load_model_class(known.model_name)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, default=known.model_name)
    parser.add_argument(
        "--dataset",
        choices=("christian", "android", "douban", "twitter", "memetracker"),
        default="christian",
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WWW_ROOT / "checkpoints" / "buzzbloom_temporal.pt",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print epoch progress but omit the full final JSON payload",
    )
    parser = Model.parse_model_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    return _base_parser().parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "minimum_epochs": args.minimum_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_prefix_length": args.max_prefix_length,
        "valid_environments": args.valid_environments,
        "d_model": args.d_model,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.steps_per_epoch < 0 or args.max_eval_batches < 0:
        raise ValueError("step and evaluation caps cannot be negative")
    if args.minimum_epochs > args.epochs:
        raise ValueError("minimum_epochs cannot exceed epochs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    seed_everything(args.seed)
    device = select_device(args.device)
    args.device = device

    loader = TemporalBuzzLoader(
        args.dataset,
        args.dataset_root,
        max_prefix_length=args.max_prefix_length,
        valid_environments=args.valid_environments,
    )
    collate = make_collate_fn(args.max_prefix_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        loader.train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    validation_loaders = [
        DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        for dataset in loader.valid_sets
    ]
    if loader.test_materialized:
        raise RuntimeError("test data was unexpectedly materialized")

    Model = load_model_class(args.model_name)
    model = Model(args, loader).to(device)
    semantics_preserving_patches = apply_semantics_preserving_patches(model, args.model_name)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[EpochResult] = []
    best_score = float("-inf")
    selected_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            steps_per_epoch=args.steps_per_epoch,
            gradient_clip=args.gradient_clip,
        )
        validation, by_environment = evaluate_validation(
            model,
            validation_loaders,
            loader.valid_environment_names,
            device,
            max_eval_batches=args.max_eval_batches,
        )
        history.append(EpochResult(epoch, training_loss, validation, by_environment))
        score = validation["map@100"]
        print(
            f"epoch={epoch:03d} loss={training_loss:.6f} "
            f"valid_map@100={score:.6f} worst={validation['worst_map@100']:.6f}",
            flush=True,
        )
        if score > best_score + args.minimum_delta:
            best_score = score
            selected_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "selected_epoch": epoch,
                    "validation_map@100": score,
                    "protocol": "temporal_70_10_20_ties_preserved_validation_only",
                },
                args.checkpoint,
            )
        else:
            stale_epochs += 1
        if epoch >= args.minimum_epochs and stale_epochs >= args.early_stopping_patience:
            break

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    restored_validation, restored_by_environment = evaluate_validation(
        model,
        validation_loaders,
        loader.valid_environment_names,
        device,
        max_eval_batches=args.max_eval_batches,
    )

    result: dict[str, Any] = {
        "model_name": args.model_name,
        "integration": "BuzzBloom community implementation",
        "dataset": args.dataset,
        "seed": args.seed,
        "device": str(device),
        "selected_epoch": selected_epoch,
        "parameter_count": parameter_count,
        "semantics_preserving_patches": semantics_preserving_patches,
        "checkpoint": str(args.checkpoint.resolve()),
        "validation": restored_validation,
        "validation_by_environment": restored_by_environment,
        "history": [
            {
                "epoch": item.epoch,
                "training_loss": item.training_loss,
                "validation": item.validation,
                "validation_by_environment": item.validation_by_environment,
            }
            for item in history
        ],
        "counts": loader.counts,
        "protocol": {
            "chronological_split": [0.7, 0.1, 0.2],
            "timestamp_ties_preserved": True,
            "train_graph_records": "train_only",
            "validation_environments": args.valid_environments,
            "selection_metric": "mean validation MAP@100",
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "selection_changes_permitted": False,
            "postfreeze_descriptive_baseline": True,
            "special_tokens_excluded_from_metrics": True,
            "validation_cascade_indices_are_dummy": True,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "steps_per_epoch": args.steps_per_epoch,
        },
        "model_arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"device", "dataset_root", "checkpoint", "result_json"}
            and isinstance(value, (str, int, float, bool, type(None), list))
        },
    }
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
