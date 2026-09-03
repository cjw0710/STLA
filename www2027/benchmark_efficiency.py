"""Benchmark frozen anchor, adaptive residual, and hierarchical fusion costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import torch

from .evaluate_checkpoint import file_sha256
from .metrics import protected_union_scores
from .models import TemporalDiffusionModel
from .train_temporal import (
    batch_to_device,
    build_prepared_protocol,
    forward_environment,
    make_loader,
    select_device,
)


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
    parser.add_argument("--benchmark-batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90_index = max(0, min(len(ordered) - 1, int(0.9 * len(ordered)) - 1))
    return {
        "mean_ms": fmean(values),
        "median_ms": median(values),
        "p90_ms": ordered[p90_index],
    }


def memory_distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean_bytes": fmean(values),
        "median_bytes": median(values),
        "max_bytes": max(values),
    }


def reset_peak_memory(device: torch.device) -> int:
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def peak_memory_measurement(
    device: torch.device, baseline_bytes: int
) -> tuple[int, int]:
    synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return max(0, peak_bytes - baseline_bytes), peak_bytes


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.benchmark_batches < 1 or args.repeats < 1:
        raise ValueError("benchmark batches and repeats must be positive")
    checkpoint_path = args.checkpoint.resolve()
    result_path = args.result_json.resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("checkpoint_sha256") == checkpoint_hash
            and existing.get("protocol", {}).get("test_materialized") is False
        ):
            print(f"efficiency_already_benchmarked result={result_path}")
            return
        raise FileExistsError(f"refusing to overwrite unrelated result: {result_path}")

    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_nodes, _, valid_cpu, _, _, _ = build_prepared_protocol(args)
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
    model.eval()
    environments = [environment.graph_to(device) for environment in valid_cpu]
    materialized = []
    for environment in environments:
        for batch in make_loader(environment, args.batch_size, shuffle=False, seed=0):
            materialized.append((environment, batch_to_device(batch, device)))
            if len(materialized) >= args.benchmark_batches:
                break
        if len(materialized) >= args.benchmark_batches:
            break
    if not materialized:
        raise RuntimeError("validation produced no benchmark batches")

    original_prior_mode = model.prior_mode
    warm_environment, warm_batch = materialized[0]
    for _ in range(args.warmup):
        model.prior_mode = "none"
        warm_anchor_output = forward_environment(model, warm_environment, warm_batch)
        del warm_anchor_output
        model.prior_mode = original_prior_mode
        warm_adaptive_output = forward_environment(model, warm_environment, warm_batch)
        warm_fused_scores = protected_union_scores(
            warm_adaptive_output.logits,
            warm_adaptive_output.base_logits,
            warm_environment.popularity_groups,
            warm_environment.recency_groups,
            topk=100,
            protected_cutoffs=(10, 50, 100),
        )
        del warm_fused_scores, warm_adaptive_output
    synchronize(device)

    timings = {"anchor_forward": [], "adaptive_forward": [], "fusion": []}
    batch_examples: list[int] = []
    for _ in range(args.repeats):
        for environment, batch in materialized:
            batch_examples.append(int(batch["target"].shape[0]))
            model.prior_mode = "none"
            synchronize(device)
            started = perf_counter()
            anchor_output = forward_environment(model, environment, batch)
            synchronize(device)
            timings["anchor_forward"].append((perf_counter() - started) * 1000)
            del anchor_output

            model.prior_mode = original_prior_mode
            synchronize(device)
            started = perf_counter()
            output = forward_environment(model, environment, batch)
            synchronize(device)
            timings["adaptive_forward"].append((perf_counter() - started) * 1000)

            started = perf_counter()
            fused_scores = protected_union_scores(
                output.logits,
                output.base_logits,
                environment.popularity_groups,
                environment.recency_groups,
                topk=100,
                protected_cutoffs=(10, 50, 100),
            )
            synchronize(device)
            timings["fusion"].append((perf_counter() - started) * 1000)
            del fused_scores, output

    memory_deltas: dict[str, list[int]] = {
        "anchor_forward": [],
        "adaptive_forward": [],
        "fusion": [],
        "adaptive_plus_fusion": [],
    }
    memory_peaks: dict[str, list[int]] = {
        name: [] for name in memory_deltas
    }
    memory_baselines: list[int] = []
    if device.type == "cuda":
        for _ in range(args.repeats):
            for environment, batch in materialized:
                model.prior_mode = "none"
                baseline_bytes = reset_peak_memory(device)
                memory_baselines.append(baseline_bytes)
                anchor_output = forward_environment(model, environment, batch)
                delta_bytes, peak_bytes = peak_memory_measurement(
                    device, baseline_bytes
                )
                memory_deltas["anchor_forward"].append(delta_bytes)
                memory_peaks["anchor_forward"].append(peak_bytes)
                del anchor_output

                model.prior_mode = original_prior_mode
                baseline_bytes = reset_peak_memory(device)
                output = forward_environment(model, environment, batch)
                delta_bytes, peak_bytes = peak_memory_measurement(
                    device, baseline_bytes
                )
                memory_deltas["adaptive_forward"].append(delta_bytes)
                memory_peaks["adaptive_forward"].append(peak_bytes)

                baseline_bytes = reset_peak_memory(device)
                fused_scores = protected_union_scores(
                    output.logits,
                    output.base_logits,
                    environment.popularity_groups,
                    environment.recency_groups,
                    topk=100,
                    protected_cutoffs=(10, 50, 100),
                )
                delta_bytes, peak_bytes = peak_memory_measurement(
                    device, baseline_bytes
                )
                memory_deltas["fusion"].append(delta_bytes)
                memory_peaks["fusion"].append(peak_bytes)
                del fused_scores, output

                baseline_bytes = reset_peak_memory(device)
                end_to_end_output = forward_environment(model, environment, batch)
                end_to_end_scores = protected_union_scores(
                    end_to_end_output.logits,
                    end_to_end_output.base_logits,
                    environment.popularity_groups,
                    environment.recency_groups,
                    topk=100,
                    protected_cutoffs=(10, 50, 100),
                )
                delta_bytes, peak_bytes = peak_memory_measurement(
                    device, baseline_bytes
                )
                memory_deltas["adaptive_plus_fusion"].append(delta_bytes)
                memory_peaks["adaptive_plus_fusion"].append(peak_bytes)
                del end_to_end_scores, end_to_end_output
    model.prior_mode = original_prior_mode

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    residual_parameters = sum(
        parameter.numel() for parameter in model.temporal_prior_gate.parameters()
    )
    summarized = {name: distribution(values) for name, values in timings.items()}
    summarized["adaptive_plus_fusion"] = distribution(
        [
            adaptive_ms + fusion_ms
            for adaptive_ms, fusion_ms in zip(
                timings["adaptive_forward"], timings["fusion"]
            )
        ]
    )
    memory = None
    if device.type == "cuda":
        memory = {
            "metric": "cuda_max_memory_allocated_bytes",
            "resident_baseline": memory_distribution(memory_baselines),
            "incremental_peak_above_stage_baseline": {
                name: memory_distribution(values)
                for name, values in memory_deltas.items()
            },
            "absolute_peak": {
                name: memory_distribution(values)
                for name, values in memory_peaks.items()
            },
        }
    result = {
        "status": "postfreeze_validation_efficiency_benchmark",
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": checkpoint["epoch"],
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "parameters": {
            "anchor": total_parameters - residual_parameters,
            "residual_gate": residual_parameters,
            "adaptive_total": total_parameters,
            "residual_fraction": residual_parameters / total_parameters,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        },
        "timing": summarized,
        "memory": memory,
        "measurements_per_stage": len(timings["anchor_forward"]),
        "mean_batch_examples": fmean(batch_examples),
        "protocol": {
            "validation_batches": len(materialized),
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_prefix_length": args.max_prefix_length,
            "test_materialized": False,
            "selection_changes_permitted": False,
            "timing_excludes_data_loading": True,
            "memory_excludes_data_loading": True,
            "memory_metric": (
                "absolute cuda_max_memory_allocated and delta from "
                "stage-start memory_allocated"
                if device.type == "cuda"
                else None
            ),
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    memory_text = (
        f"peak_memory_mib="
        f"{memory['absolute_peak']['adaptive_plus_fusion']['max_bytes'] / 2**20:.3f} "
        f"incremental_peak_mib="
        f"{memory['incremental_peak_above_stage_baseline']['adaptive_plus_fusion']['max_bytes'] / 2**20:.3f} "
        if memory is not None
        else ""
    )
    print(
        f"efficiency dataset={args.dataset} device={result['device']} "
        f"anchor_ms={summarized['anchor_forward']['mean_ms']:.3f} "
        f"adaptive_ms={summarized['adaptive_forward']['mean_ms']:.3f} "
        f"fusion_ms={summarized['fusion']['mean_ms']:.3f} "
        f"{memory_text}"
        f"result={result_path}"
    )


if __name__ == "__main__":
    main()
