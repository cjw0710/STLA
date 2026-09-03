"""Report exact post-loader counts for the chronological WWW protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_temporal import build_prepared_protocol, prepare_final_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["christian", "android", "douban", "twitter"],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument("--train-environments", type=int, default=4)
    parser.add_argument("--valid-environments", type=int, default=2)
    parser.add_argument("--test-environments", type=int, default=3)
    parser.add_argument("--sample-hop", type=int, default=2)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def active_users(records) -> int:
    return len({node for record in records for node in record.cascade})


def split_counts(records, environments) -> dict[str, object]:
    return {
        "cascades": len(records),
        "active_users": active_users(records),
        "prefix_next_user_examples": sum(
            len(environment.dataset) for environment in environments
        ),
        "target_environment_cascades": [
            len(environment.dataset.records) for environment in environments
        ],
        "target_environment_examples": [
            len(environment.dataset) for environment in environments
        ],
        "past_graph_directed_edges": [
            int(environment.edge_index.shape[1]) for environment in environments
        ],
    }


def main() -> None:
    args = parse_args()
    datasets: dict[str, object] = {}
    for dataset in args.datasets:
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset
        (
            num_nodes,
            train_environments,
            valid_environments,
            train_records,
            valid_records,
            test_records,
        ) = build_prepared_protocol(dataset_args)
        test_environments = prepare_final_test(
            dataset_args,
            num_nodes,
            train_records,
            valid_records,
            test_records,
        )
        all_records = (*train_records, *valid_records, *test_records)
        datasets[dataset] = {
            "node_vocabulary": num_nodes,
            "observed_active_users": active_users(all_records),
            "cascades": len(all_records),
            "raw_activations": sum(len(record.cascade) for record in all_records),
            "train": split_counts(train_records, train_environments),
            "validation": split_counts(valid_records, valid_environments),
            "test": split_counts(test_records, test_environments),
        }
        print(
            f"{dataset} users={num_nodes} cascades={len(all_records)} "
            f"examples={sum(len(environment.dataset) for environment in (*train_environments, *valid_environments, *test_environments))}"
        )

    result = {
        "status": "descriptive_protocol_audit_after_frozen_one_shot",
        "split": "chronological_70_10_20_timestamp_ties_preserved",
        "max_prefix_length": args.max_prefix_length,
        "sample_hop": args.sample_hop,
        "environment_counts": {
            "train": args.train_environments,
            "validation": args.valid_environments,
            "test": args.test_environments,
        },
        "datasets": datasets,
        "test_statistics_materialized": True,
        "selection_changes_permitted": False,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
