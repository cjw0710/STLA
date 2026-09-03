"""Attach the temporal safety adapter directly to a frozen CIKM DeDiff model.

The anchor is the unchanged model/loss trained by ``train_temporal_dediff``
under the corrected chronological protocol.  Only the small logit adapter is
optimized here; checkpoint selection is validation-only and test cascades are
never materialized as a Dataset, DataLoader, tensor, or model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import (
    build_rolling_snapshots,
    chronological_split,
    load_cascades,
    make_temporal_environments,
)
from .metrics import RankingAccumulator, aggregate_environment_metrics, protected_union_scores
from .models import TemporalLogitAdapter
from .train_temporal_dediff import (
    CUTOFFS,
    DeDiff,
    DeDiffEnvironment,
    make_loaders,
    prepare_protocol,
)
from .training import PreparedEnvironment, prepare_environment


ROOT = Path(__file__).resolve().parent
PATHS = ("anchor", "adaptive", "hierarchical_union")


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
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--environment-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--node-rank", type=int, default=0)
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


def _audit_anchor(anchor_result: dict[str, Any]) -> None:
    protocol = anchor_result.get("protocol", {})
    fields = ("test_materialized", "test_evaluated", "test_used_for_selection")
    if any(protocol.get(field) is not False for field in fields):
        raise ValueError("DeDiff anchor did not pass the validation-only protocol audit")
    if anchor_result.get("status") != "validation_only_corrected_dediff":
        raise ValueError("anchor result is not a corrected temporal DeDiff run")


def _anchor_arguments(
    anchor_result: dict[str, Any],
    dataset_root: Path,
) -> SimpleNamespace:
    values = dict(anchor_result["model_arguments"])
    values["dataset_root"] = dataset_root
    values["checkpoint"] = Path(values["checkpoint"])
    values["result_json"] = Path(values["result_json"])
    return SimpleNamespace(**values)


def build_contexts(
    anchor_args: SimpleNamespace,
    *,
    num_nodes: int,
) -> tuple[list[PreparedEnvironment], list[PreparedEnvironment]]:
    records = load_cascades(anchor_args.dataset_root / anchor_args.dataset)
    split = chronological_split(records)
    train_groups = make_temporal_environments(
        split.train,
        anchor_args.train_environments,
        prefix="train",
    )
    valid_groups = make_temporal_environments(
        split.valid,
        anchor_args.valid_environments,
        prefix="valid",
    )
    train_snapshots = build_rolling_snapshots(train_groups, num_nodes, sample_hop=2)
    valid_snapshots = build_rolling_snapshots(
        valid_groups,
        num_nodes,
        warm_start_records=split.train,
        warm_start_recent_records=train_groups[-1].records,
        sample_hop=2,
    )
    return (
        [
            prepare_environment(snapshot, num_nodes, anchor_args.max_prefix_length)
            for snapshot in train_snapshots
        ],
        [
            prepare_environment(snapshot, num_nodes, anchor_args.max_prefix_length)
            for snapshot in valid_snapshots
        ],
    )


def _move_contexts(
    contexts: Sequence[PreparedEnvironment],
    device: torch.device,
) -> list[PreparedEnvironment]:
    return [context.graph_to(device) for context in contexts]


def load_frozen_anchor(
    anchor_result: dict[str, Any],
    anchor_args: SimpleNamespace,
    model_args: SimpleNamespace,
) -> DeDiff:
    anchor = DeDiff(model_args).to(model_args.device)
    checkpoint = torch.load(
        Path(anchor_result["checkpoint"]),
        map_location=model_args.device,
        weights_only=False,
    )
    anchor.load_state_dict(checkpoint["model_state"])
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    if int(checkpoint["selected_epoch"]) != int(anchor_result["selected_epoch"]):
        raise ValueError("anchor checkpoint/result epoch mismatch")
    return anchor


def _anchor_logits(
    anchor: DeDiff,
    model_args: SimpleNamespace,
    batch: dict[str, torch.Tensor],
    environment: DeDiffEnvironment,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        prediction, label, _ = anchor(model_args, batch, environment.info)
    # DeDiff reserves column 0 for padding and the last column for EOS.  The
    # common evaluation vocabulary uses original user ids, including an
    # impossible candidate zero for exact index alignment.
    logits = torch.cat(
        [prediction.new_full((prediction.shape[0], 1), float("-inf")), prediction[:, 1:-1]],
        dim=1,
    )
    return logits, label


def train_epoch(
    anchor: DeDiff,
    adapter: TemporalLogitAdapter,
    model_args: SimpleNamespace,
    environments: Sequence[DeDiffEnvironment],
    contexts: Sequence[PreparedEnvironment],
    loaders: Sequence[DataLoader],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    gradient_clip: float,
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
        anchor_logits, label = _anchor_logits(
            anchor,
            model_args,
            batch,
            environments[environment_index],
        )
        context = contexts[environment_index]
        sequence = batch["cascade"].to(model_args.device, non_blocking=True)
        timestamp = batch["timestamp"].to(model_args.device, non_blocking=True)
        adaptive_logits, _ = adapter(
            anchor_logits,
            sequence,
            timestamp,
            context.environment_features,
            context.historical_popularity,
            context.recent_popularity,
            input_id_offset=0,
        )
        valid = label.ne(0) & label.ne(model_args.user_num - 1)
        loss = torch.nn.functional.cross_entropy(adaptive_logits[valid], label[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(adapter.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / steps


@torch.no_grad()
def evaluate(
    anchor: DeDiff,
    adapter: TemporalLogitAdapter,
    model_args: SimpleNamespace,
    environments: Sequence[DeDiffEnvironment],
    contexts: Sequence[PreparedEnvironment],
    loaders: Sequence[DataLoader],
    *,
    max_batches: int,
) -> dict[str, Any]:
    anchor.eval()
    adapter.eval()
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
            anchor_all, label = _anchor_logits(anchor, model_args, batch, environment)
            sequence = batch["cascade"].to(model_args.device, non_blocking=True)
            timestamp = batch["timestamp"].to(model_args.device, non_blocking=True)
            adaptive_all, _ = adapter(
                anchor_all,
                sequence,
                timestamp,
                context.environment_features,
                context.historical_popularity,
                context.recent_popularity,
                input_id_offset=0,
            )
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
        raise RuntimeError("the unchanged CIKM DeDiff implementation requires CUDA")
    anchor_result = json.loads(args.anchor_result.read_text(encoding="utf-8"))
    _audit_anchor(anchor_result)
    seed = int(anchor_result["seed"])
    seed_everything(seed)
    anchor_args = _anchor_arguments(anchor_result, args.dataset_root)
    model_args, train_environments, valid_environments, _, num_nodes = prepare_protocol(anchor_args)
    train_contexts, valid_contexts = build_contexts(anchor_args, num_nodes=num_nodes)
    train_contexts = _move_contexts(train_contexts, model_args.device)
    valid_contexts = _move_contexts(valid_contexts, model_args.device)
    if [item.name for item in train_environments] != [item.name for item in train_contexts]:
        raise RuntimeError("training environment alignment failed")
    if [item.name for item in valid_environments] != [item.name for item in valid_contexts]:
        raise RuntimeError("validation environment alignment failed")

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
    adapter = TemporalLogitAdapter(
        num_nodes,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        environment_hidden_dim=args.environment_hidden_dim,
        dropout=args.dropout,
        node_rank=args.node_rank,
    ).to(model_args.device)
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
        model_args,
        valid_environments,
        valid_contexts,
        valid_loaders,
        max_batches=args.max_eval_batches,
    )
    best_score = initial["paths"]["hierarchical_union"]["metrics"]["map@100"]
    selected_epoch = 0
    torch.save({"adapter_state": adapter.state_dict(), "selected_epoch": 0}, args.checkpoint)
    history: list[dict[str, Any]] = [{"epoch": 0, "training_loss": None, "evaluation": initial}]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            anchor,
            adapter,
            model_args,
            train_environments,
            train_contexts,
            train_loaders,
            optimizer,
            steps=args.steps_per_epoch,
            gradient_clip=args.gradient_clip,
        )
        evaluation = evaluate(
            anchor,
            adapter,
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
                {"adapter_state": adapter.state_dict(), "selected_epoch": epoch},
                args.checkpoint,
            )
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    selected = torch.load(args.checkpoint, map_location=model_args.device, weights_only=False)
    adapter.load_state_dict(selected["adapter_state"])
    final = evaluate(
        anchor,
        adapter,
        model_args,
        valid_environments,
        valid_contexts,
        valid_loaders,
        max_batches=args.max_eval_batches,
    )
    result = {
        "status": "validation_only_direct_dediff_adapter",
        "dataset": anchor_result["dataset"],
        "seed": seed,
        "anchor_model": "unchanged CIKM DeDiff",
        "anchor_result": str(args.anchor_result.resolve()),
        "anchor_checkpoint": anchor_result["checkpoint"],
        "selected_epoch": selected_epoch,
        "adapter_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "adapter_node_rank": args.node_rank,
        "evaluation": final,
        "history": history,
        "protocol": {
            "relationship_to_cikm": "direct frozen-logit extension of unchanged DeDiff",
            "chronological_split": [0.7, 0.1, 0.2],
            "train_environments": anchor_args.train_environments,
            "validation_environments": anchor_args.valid_environments,
            "past_only_adapter_features": True,
            "anchor_frozen": True,
            "selection_metric": "hierarchical union mean validation MAP@100",
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "selection_changes_permitted": False,
            "postfreeze_descriptive": True,
        },
    }
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "adapter_parameter_count": result["adapter_parameter_count"],
                "metrics": final["paths"]["hierarchical_union"]["metrics"],
                "guarantee": final["guarantee"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
