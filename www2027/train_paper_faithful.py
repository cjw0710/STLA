"""Train the formula-faithful DeDiff candidate without reading test data.

This development runner reads only the released ``cascade_train.json`` and
``cascade_valid.json`` files.  It remaps the legacy one-based real-user ids to
zero-based ids, builds both graph inputs from training-period information, and
selects checkpoints only by validation MAP@100.  It never opens
``cascade_test.json`` and therefore cannot produce a test metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import load_npz
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import CascadeRecord, NextUserDataset, build_interaction_graph, load_cascades, popularity_counts
from .metrics import RankingAccumulator
from .models import PaperFaithfulDeDiff, paper_dediff_loss


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "config" / "paper_faithful_v1.json"
CUTOFFS = (10, 50, 100)


class PaperPrefixDataset(NextUserDataset):
    """Add pairwise social distances to strict prefix/next-user examples."""

    def __init__(
        self,
        records: Sequence[CascadeRecord],
        num_nodes: int,
        max_prefix_length: int,
        social_distance: np.ndarray,
    ) -> None:
        super().__init__(records, num_nodes, max_prefix_length)
        if social_distance.shape != (num_nodes, num_nodes):
            raise ValueError("social_distance must match the remapped real-user vocabulary")
        self.social_distance = social_distance

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = super().__getitem__(index)
        length = int(example["length"])
        users = example["prefix"][:length].numpy()
        pairwise = self.social_distance[np.ix_(users, users)]
        padded = torch.full(
            (self.max_prefix_length, self.max_prefix_length),
            float(self.num_nodes + 1),
            dtype=torch.float32,
        )
        padded[:length, :length] = torch.as_tensor(pairwise, dtype=torch.float32)
        example["social_distance"] = padded
        return example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("twitter", "douban", "android", "christian"), default="christian")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 uses the full train loader")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="nonzero is engineering smoke only")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prefix-length", type=int, default=200)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gcn-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=10)
    parser.add_argument("--attention-head-dimension", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-disagreement", type=float, default=1.0)
    parser.add_argument("--lambda-inter-view", type=float, default=1.0)
    parser.add_argument(
        "--kl-direction",
        choices=("prediction_to_target", "target_to_prediction"),
        default="prediction_to_target",
    )
    parser.add_argument("--hinged-disagreement", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "www2027" / "checkpoints" / "paper_faithful_smoke.pt",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=REPO_ROOT / "www2027" / "artifacts" / "paper_faithful_smoke.json",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remap_one_based_records(records: Iterable[CascadeRecord]) -> tuple[CascadeRecord, ...]:
    """Remove the legacy PAD offset while preserving provenance and times."""

    remapped = []
    for record in records:
        if any(node < 1 for node in record.cascade):
            raise ValueError("the released DeDiff bundle is expected to use one-based real-user ids")
        remapped.append(
            CascadeRecord(
                cascade=tuple(node - 1 for node in record.cascade),
                timestamp=record.timestamp,
                source_split=record.source_split,
                source_index=record.source_index,
            )
        )
    return tuple(remapped)


def scipy_graph_tensors(matrix, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    graph = matrix.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([graph.row, graph.col]).astype(np.int64, copy=False)
    ).to(device)
    edge_weight = torch.from_numpy(graph.data.astype(np.float32, copy=False)).to(device)
    return edge_index, edge_weight


def prepare_development_data(args: argparse.Namespace):
    dataset_dir = args.dataset_root / args.dataset
    # Explicit split_names ensure cascade_test.json is never opened here.
    train_records = remap_one_based_records(load_cascades(dataset_dir, split_names=("train",)))
    valid_records = remap_one_based_records(load_cascades(dataset_dir, split_names=("valid",)))
    social_raw = load_npz(dataset_dir / "social_graph.npz").tocsr()
    if social_raw.shape[0] != social_raw.shape[1] or social_raw.shape[0] < 3:
        raise ValueError("released social graph must be a square PAD/real-user/EOS matrix")
    num_nodes = social_raw.shape[0] - 2
    maximum = max(node for record in (*train_records, *valid_records) for node in record.cascade)
    if maximum >= num_nodes:
        raise ValueError("a cascade user falls outside the released social graph vocabulary")

    # Raw rows/columns 1..N are real users. Row zero is legacy PAD and the
    # final row is EOS. Make the social view binary as defined in the PDF.
    social = social_raw[1 : num_nodes + 1, 1 : num_nodes + 1].copy().tocsr()
    social.setdiag(0)
    social.eliminate_zeros()
    social.data.fill(1.0)
    interaction = build_interaction_graph(
        train_records,
        num_nodes,
        sample_hop=args.sample_hop,
        undirected=True,
        add_self_loops=False,
        binary=False,
    )
    distance_raw = np.load(dataset_dir / "distance.npy", allow_pickle=True, mmap_mode="r")
    distance = distance_raw[1 : num_nodes + 1, 1 : num_nodes + 1]
    train_dataset = PaperPrefixDataset(
        train_records,
        num_nodes,
        args.max_prefix_length,
        distance,
    )
    valid_dataset = PaperPrefixDataset(
        valid_records,
        num_nodes,
        args.max_prefix_length,
        distance,
    )
    popularity = torch.from_numpy(
        popularity_counts(train_records, num_nodes).astype(np.float32, copy=False)
    )
    return train_records, valid_records, train_dataset, valid_dataset, interaction, social, popularity


def make_loader(
    dataset: PaperPrefixDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=pin_memory,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_model(
    model: PaperFaithfulDeDiff,
    batch: dict[str, torch.Tensor],
    graph_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    hinged_disagreement: bool,
):
    interaction_edge_index, interaction_edge_weight, social_edge_index, social_edge_weight = graph_tensors
    return model(
        prefix=batch["prefix"],
        elapsed=batch["elapsed"],
        lengths=batch["length"],
        social_distance=batch["social_distance"],
        interaction_edge_index=interaction_edge_index,
        interaction_edge_weight=interaction_edge_weight,
        social_edge_index=social_edge_index,
        social_edge_weight=social_edge_weight,
        hinged_disagreement=hinged_disagreement,
    )


def train_epoch(
    model: PaperFaithfulDeDiff,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    popularity: torch.Tensor,
    graph_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sums = {key: 0.0 for key in ("total", "prediction", "bias", "disagreement", "inter_view")}
    steps = 0
    for batch in loader:
        if args.steps_per_epoch and steps >= args.steps_per_epoch:
            break
        batch = move_batch(batch, device)
        output = forward_model(
            model,
            batch,
            graph_tensors,
            hinged_disagreement=args.hinged_disagreement,
        )
        breakdown = paper_dediff_loss(
            output,
            batch["target"],
            popularity,
            alpha=args.alpha,
            lambda_disagreement=args.lambda_disagreement,
            lambda_inter_view=args.lambda_inter_view,
            kl_direction=args.kl_direction,
        )
        if not bool(torch.isfinite(breakdown.total)):
            raise FloatingPointError("paper-faithful loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()
        clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        for key in sums:
            sums[key] += float(getattr(breakdown, key).detach())
        steps += 1
    if steps == 0:
        raise RuntimeError("no training batches were consumed")
    return {key: value / steps for key, value in sums.items()}


@torch.no_grad()
def evaluate(
    model: PaperFaithfulDeDiff,
    loader: DataLoader,
    graph_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    hinged_disagreement: bool,
    max_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = RankingAccumulator(CUTOFFS)
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        output = forward_model(
            model,
            batch,
            graph_tensors,
            hinged_disagreement=hinged_disagreement,
        )
        accumulator.update(output.logits, batch["target"])
    return accumulator.compute()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.minimum_epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, minimum_epochs, patience, and batch_size must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    (
        train_records,
        valid_records,
        train_dataset,
        valid_dataset,
        interaction,
        social,
        popularity,
    ) = prepare_development_data(args)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    graph_tensors = (
        *scipy_graph_tensors(interaction, device),
        *scipy_graph_tensors(social, device),
    )
    popularity = popularity.to(device)
    model = PaperFaithfulDeDiff(
        num_nodes=interaction.shape[0],
        dimension=args.dimension,
        rank=args.rank,
        gcn_layers=args.gcn_layers,
        attention_heads=args.attention_heads,
        attention_head_dimension=args.attention_head_dimension,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_map = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            popularity,
            graph_tensors,
            args,
            device,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            graph_tensors,
            hinged_disagreement=args.hinged_disagreement,
            max_batches=args.max_eval_batches,
            device=device,
        )
        history.append({"epoch": epoch, "train": train_loss, "validation": valid_metrics})
        print(
            f"epoch={epoch} loss={train_loss['total']:.6f} "
            f"validation_map@100={valid_metrics['map@100']:.6f}"
        )
        if valid_metrics["map@100"] > best_map:
            best_map = valid_metrics["map@100"]
            best_epoch = epoch
            best_metrics = valid_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("validation selection failed to produce a checkpoint")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "dataset": args.dataset,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        },
        args.checkpoint,
    )
    result = {
        "status": "engineering_smoke" if args.max_eval_batches or args.steps_per_epoch else "validation_only",
        "model": "paper_formula_faithful_dediff_candidate",
        "dataset": args.dataset,
        "seed": args.seed,
        "device": str(device),
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "test_file_opened": False,
        "test_metrics": None,
        "train_cascades": len(train_records),
        "validation_cascades": len(valid_records),
        "train_examples": len(train_dataset),
        "validation_examples": len(valid_dataset),
        "num_real_users": interaction.shape[0],
        "interaction_edges": int(interaction.nnz),
        "social_edges": int(social.nnz),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "configuration": vars(args),
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
        "history": history,
        "checkpoint": str(args.checkpoint),
    }
    # Convert Paths before JSON serialization without mutating argparse state.
    result["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result["configuration"].items()
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "validation": best_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
