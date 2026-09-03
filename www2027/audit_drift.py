"""Command-line audit of temporal popularity drift in DeDiff datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    chronological_split,
    load_cascades,
    make_temporal_environments,
    popularity_counts,
)
from .metrics import compute_drift_report


def audit_dataset(
    dataset_dir: Path,
    windows: int,
    hub_fraction: float,
    scope: str = "full",
) -> dict[str, object]:
    all_records = load_cascades(dataset_dir)
    if scope == "full":
        records = all_records
    else:
        split = chronological_split(all_records)
        if scope == "train":
            records = split.train
        elif scope == "train_valid":
            records = (*split.train, *split.valid)
        else:
            raise ValueError("scope must be full, train, or train_valid")
    environments = make_temporal_environments(records, windows, prefix="window")
    num_nodes = max(node for record in all_records for node in record.cascade) + 1
    populations = [popularity_counts(environment.records, num_nodes) for environment in environments]
    report = compute_drift_report(populations, hub_fraction=hub_fraction)
    return {
        "dataset": dataset_dir.name,
        "scope": scope,
        "cascades": len(records),
        "total_cascades": len(all_records),
        "num_nodes": num_nodes,
        "windows": windows,
        "hub_fraction": hub_fraction,
        "window_sizes": [len(environment.records) for environment in environments],
        **report.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--hub-fraction", type=float, default=0.2)
    parser.add_argument(
        "--scope",
        choices=("full", "train", "train_valid"),
        default="full",
        help="Restrict the audit before temporal windows are constructed",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = args.datasets or sorted(
        path.name for path in args.dataset_root.iterdir() if path.is_dir()
    )
    results = [
        audit_dataset(
            args.dataset_root / dataset,
            args.windows,
            args.hub_fraction,
            args.scope,
        )
        for dataset in dataset_names
    ]

    header = f"{'dataset':<12} {'cascades':>8} {'users':>8} {'JSD':>9} {'hub J':>9} {'churn':>9}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['dataset']:<12} {result['cascades']:>8} {result['num_nodes']:>8} "
            f"{result['mean_js_divergence']:>9.3f} "
            f"{result['mean_top_hub_jaccard']:>9.3f} "
            f"{result['mean_active_user_churn']:>9.3f}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
