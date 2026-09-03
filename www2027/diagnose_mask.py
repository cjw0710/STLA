"""Inspect whether a trained low-rank mask actually changes across environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import build_rolling_snapshots, chronological_split, load_cascades, make_temporal_environments
from .models import TemporalDiffusionModel
from .training import prepare_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    records = load_cascades(args.dataset_root / args.dataset)
    split = chronological_split(records)
    num_nodes = max(node for record in records for node in record.cascade) + 1
    environments = make_temporal_environments(
        split.train, args.environments, prefix="train"
    )
    snapshots = build_rolling_snapshots(
        environments,
        num_nodes,
        sample_hop=args.sample_hop,
    )
    features = torch.stack(
        [
            prepare_environment(snapshot, num_nodes, max_prefix_length=2).environment_features
            for snapshot in snapshots
        ]
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    configuration = checkpoint["model_config"]
    model = TemporalDiffusionModel(num_nodes=num_nodes, **configuration).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if unexpected or any(
        not name.startswith("temporal_prior_gate.") for name in missing
    ):
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    contexts = model.environment_encoder(features)
    common_nodes = torch.arange(num_nodes, device=device)
    common_edges = torch.stack([common_nodes, common_nodes])
    masks: list[torch.Tensor] = []
    scales: list[dict[str, float]] = []
    for context in contexts:
        mask_context = torch.zeros_like(context) if model.mask_mode == "static" else context
        mask = model.graph_mask(common_edges, mask_context)
        masks.append(mask)
        left_scale = 1.0 + torch.tanh(model.graph_mask.left_scale(mask_context))
        right_scale = 1.0 + torch.tanh(model.graph_mask.right_scale(mask_context))
        scales.append(
            {
                "left_mean": float(left_scale.mean()),
                "left_std": float(left_scale.std(unbiased=False)),
                "right_mean": float(right_scale.mean()),
                "right_std": float(right_scale.std(unbiased=False)),
            }
        )

    mask_tensor = torch.stack(masks)
    consecutive_context_l2 = torch.linalg.vector_norm(contexts[1:] - contexts[:-1], dim=1)
    consecutive_mask_l1 = torch.mean(torch.abs(mask_tensor[1:] - mask_tensor[:-1]), dim=1)
    report = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "mask_mode": model.mask_mode,
        "selected_epoch": checkpoint["epoch"],
        "context_norm": torch.linalg.vector_norm(contexts, dim=1).cpu().tolist(),
        "consecutive_context_l2": consecutive_context_l2.cpu().tolist(),
        "mask_mean": mask_tensor.mean(dim=1).cpu().tolist(),
        "mask_std_across_edges": mask_tensor.std(dim=1, unbiased=False).cpu().tolist(),
        "consecutive_mask_mean_absolute_change": consecutive_mask_l1.cpu().tolist(),
        "scales": scales,
    }
    return report


def main() -> None:
    args = parse_args()
    report = diagnose(args)
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
