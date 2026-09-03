"""Benchmark the original and associative sparse DeDiff validation forwards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .train_dediff_logit_adapter import (
    _anchor_arguments,
    _audit_anchor,
    build_contexts,
    load_frozen_anchor,
)
from .train_dynamic_dediff_adapter import attach_contexts, load_dynamic_model
from .train_temporal_dediff import make_loaders, prepare_protocol


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--temporal-rank", type=int, default=8)
    parser.add_argument("--base-rank", type=int, default=0)
    parser.add_argument("--temporal-hidden-dim", type=int, default=32)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def benchmark(model, model_args, batch, info, *, warmup: int, repetitions: int) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        model(model_args, batch, info)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = perf_counter()
    for _ in range(repetitions):
        model(model_args, batch, info)
    end_event.record()
    torch.cuda.synchronize()
    wall_elapsed = perf_counter() - wall_start
    return {
        "mean_cuda_ms": float(start_event.elapsed_time(end_event) / repetitions),
        "mean_wall_ms": float(wall_elapsed * 1000.0 / repetitions),
        "incremental_peak_mib": float(
            (torch.cuda.max_memory_allocated() - baseline_allocated) / (1024.0**2)
        ),
    }


def main() -> None:
    args = parse_args()
    anchor_result = json.loads(args.anchor_result.read_text(encoding="utf-8"))
    _audit_anchor(anchor_result)
    anchor_args = _anchor_arguments(anchor_result, args.dataset_root)
    model_args, train_environments, valid_environments, _, num_nodes = prepare_protocol(anchor_args)
    train_contexts, valid_contexts = build_contexts(anchor_args, num_nodes=num_nodes)
    attach_contexts(train_environments, train_contexts, model_args.device)
    attach_contexts(valid_environments, valid_contexts, model_args.device)
    valid_loaders = make_loaders(
        valid_environments,
        batch_size=args.batch_size,
        shuffle=False,
        seed=int(anchor_result["seed"]),
    )
    batch = next(iter(valid_loaders[0]))
    anchor = load_frozen_anchor(anchor_result, anchor_args, model_args)
    dynamic = load_dynamic_model(anchor_result, model_args, args)
    original = benchmark(
        anchor,
        model_args,
        batch,
        valid_environments[0].info,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    associative_sparse = benchmark(
        dynamic,
        model_args,
        batch,
        valid_environments[0].info,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    result = {
        "dataset": anchor_result["dataset"],
        "seed": anchor_result["seed"],
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "original": original,
        "associative_sparse": associative_sparse,
        "base_rank": args.base_rank,
        "speedup": original["mean_cuda_ms"] / associative_sparse["mean_cuda_ms"],
        "protocol": {
            "validation_batch_only": True,
            "zero_initialized_temporal_correction": True,
            "test_materialized": False,
            "test_evaluated": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
