"""Evaluate deterministic past-only popularity baselines on validation blocks.

The baselines intentionally have no learned parameters and never construct a
test Dataset, DataLoader, tensor, or prediction.  Candidate scores for each
validation environment are computed only from preceding cascades; users
already observed in the current prefix are masked before exact ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from .baselines.buzzbloom_temporal import TemporalBuzzLoader
from .metrics import RankingAccumulator, aggregate_environment_metrics
from .train_strong_logit_adapter import build_adapter_environments, make_loaders


ROOT = Path(__file__).resolve().parent
DATASETS = ("christian", "android", "douban", "twitter")
METHODS = ("historical_popularity", "recent_popularity", "popularity_momentum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts" / "popularity_baseline_validation_summary.json",
    )
    return parser.parse_args()


def _standardize(values: torch.Tensor) -> torch.Tensor:
    values = values.float().flatten()
    deviation = values.std(unbiased=False)
    if float(deviation) <= 1e-12:
        return torch.zeros_like(values)
    return (values - values.mean()) / deviation


def _candidate_scores(environment) -> dict[str, torch.Tensor]:
    historical_log = torch.log1p(environment.historical_popularity.float())
    recent_log = torch.log1p(environment.recent_popularity.float())
    historical = _standardize(historical_log)
    recent = _standardize(recent_log)
    momentum = recent - historical
    # RankingAccumulator treats exact ties as sharing the best rank.  Add a
    # tiny, deterministic user-id tie break that cannot change unequal count
    # levels and makes every reported rank total rather than optimistic.
    tie_break = -torch.arange(historical.numel(), dtype=torch.float32)
    tie_break *= 1e-7 / max(1, historical.numel() - 1)
    return {
        "historical_popularity": historical + tie_break,
        "recent_popularity": recent + tie_break,
        "popularity_momentum": momentum + tie_break,
    }


def prefix_masked_scores(base_scores: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
    """Expand one candidate score vector and mask each observed prefix."""

    if base_scores.ndim != 1 or sequence.ndim != 2:
        raise ValueError("expected [N] scores and [B, L] shifted sequences")
    batch_size, length = sequence.shape
    if length < 2:
        raise ValueError("a ranking batch needs at least one prefix and target")
    steps = length - 1
    scores = base_scores.reshape(1, 1, -1).expand(batch_size, steps, -1).clone()
    for step in range(steps):
        observed = sequence[:, : step + 1] - 2
        valid = observed.ge(0) & observed.lt(base_scores.numel())
        if not bool(valid.any()):
            continue
        rows = torch.arange(batch_size).reshape(-1, 1).expand_as(observed)
        step_scores = scores[:, step, :]
        step_scores[rows[valid], observed[valid]] = -torch.inf
    return scores.reshape(batch_size * steps, -1)


@torch.no_grad()
def evaluate_dataset(
    dataset: str,
    *,
    dataset_root: Path,
    train_environments: int,
    valid_environments: int,
    max_prefix_length: int,
    batch_size: int,
) -> dict:
    loader = TemporalBuzzLoader(
        dataset,
        dataset_root,
        max_prefix_length=max_prefix_length,
        valid_environments=valid_environments,
    )
    _, environments = build_adapter_environments(
        loader,
        train_count=train_environments,
        valid_count=valid_environments,
        max_prefix_length=max_prefix_length,
    )
    data_loaders = make_loaders(
        environments,
        batch_size=batch_size,
        max_prefix_length=max_prefix_length,
        shuffle=False,
        seed=0,
    )
    by_method: dict[str, dict] = {}
    for method in METHODS:
        environment_metrics = []
        for environment, data_loader in zip(environments, data_loaders):
            accumulator = RankingAccumulator()
            base_scores = _candidate_scores(environment)[method]
            for sequence, _, _ in data_loader:
                gold = sequence[:, 1:].reshape(-1)
                valid = gold.ne(0)
                scores = prefix_masked_scores(base_scores, sequence)
                accumulator.update(scores[valid], gold[valid] - 2)
            environment_metrics.append(accumulator.compute())
        by_method[method] = {
            "aggregate": aggregate_environment_metrics(environment_metrics),
            "by_environment": {
                environment.name: metrics
                for environment, metrics in zip(environments, environment_metrics)
            },
        }
    return {
        "dataset": dataset,
        "methods": by_method,
        "protocol": {
            "past_only_environment_features": True,
            "observed_prefix_users_masked": True,
            "test_dataset_constructed": False,
            "test_tensor_constructed": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "validation_environments": valid_environments,
            "max_prefix_length": max_prefix_length,
            "deterministic_user_id_tie_break": True,
        },
    }


def main() -> None:
    args = parse_args()
    results = {
        dataset: evaluate_dataset(
            dataset,
            dataset_root=args.dataset_root,
            train_environments=args.train_environments,
            valid_environments=args.valid_environments,
            max_prefix_length=args.max_prefix_length,
            batch_size=args.batch_size,
        )
        for dataset in args.datasets
    }
    payload = {
        "status": "validation_only_deterministic_popularity_baselines",
        "datasets": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for dataset, result in results.items():
        for method, values in result["methods"].items():
            metrics = values["aggregate"]
            print(
                f"{dataset:12s} {method:23s} "
                f"MAP@100={metrics['map@100']:.5f} "
                f"Worst={metrics['worst_map@100']:.5f}"
            )
    print(args.output_json)


if __name__ == "__main__":
    main()
