"""Aggregate post-freeze validation-only popularity stress evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .summarize_validation_results import mean_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def summarize_dataset(by_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    seeds = sorted(by_seed)
    condition_keys = {
        (str(condition["stress"]), float(condition["severity"]))
        for item in by_seed.values()
        for condition in item["conditions"]
    }
    expected_per_seed = len(condition_keys)
    conditions: dict[str, dict[str, object]] = {}
    for stress, severity in sorted(condition_keys):
        key = f"{stress}@{severity:g}"
        selected: list[dict[str, object]] = []
        for seed in seeds:
            matches = [
                condition
                for condition in by_seed[seed]["conditions"]
                if condition["stress"] == stress
                and float(condition["severity"]) == severity
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {key} condition for seed {seed}, got {len(matches)}"
                )
            selected.append(matches[0])

        fused_minus_unperturbed = [
            float(condition["fused_minus_unperturbed"]["map@100"])
            for condition in selected
        ]
        worst_minus_unperturbed = [
            float(condition["fused_minus_unperturbed"]["worst_map@100"])
            for condition in selected
        ]
        fused_minus_anchor = [
            float(condition["fused_minus_anchor"]["map@100"])
            for condition in selected
        ]
        worst_minus_anchor = [
            float(condition["fused_minus_anchor"]["worst_map@100"])
            for condition in selected
        ]
        conditions[key] = {
            "stress": stress,
            "severity": severity,
            "fused_minus_unperturbed_map@100": mean_std(fused_minus_unperturbed),
            "fused_minus_unperturbed_worst_map@100": mean_std(
                worst_minus_unperturbed
            ),
            "fused_minus_anchor_map@100": mean_std(fused_minus_anchor),
            "fused_minus_anchor_worst_map@100": mean_std(worst_minus_anchor),
            "positive_vs_unperturbed_seeds": sum(
                delta > 0 for delta in fused_minus_unperturbed
            ),
            "positive_vs_anchor_seeds": sum(delta > 0 for delta in fused_minus_anchor),
            "guarantee": {
                "by_cutoff": {
                    cutoff: {
                        "protected_anchor_hits": sum(
                            int(
                                condition["guarantee"]["by_cutoff"][cutoff][
                                    "protected_anchor_hits"
                                ]
                            )
                            for condition in selected
                        ),
                        "violations": sum(
                            int(
                                condition["guarantee"]["by_cutoff"][cutoff][
                                    "violations"
                                ]
                            )
                            for condition in selected
                        ),
                    }
                    for cutoff in ("10", "50", "100")
                },
                "protected_anchor_hits": sum(
                    int(condition["guarantee"]["protected_anchor_hits"])
                    for condition in selected
                ),
                "violations": sum(
                    int(condition["guarantee"]["violations"])
                    for condition in selected
                ),
            },
        }

    for seed, item in by_seed.items():
        if len(item["conditions"]) != expected_per_seed:
            raise ValueError(
                f"seed {seed} has {len(item['conditions'])} conditions, "
                f"expected {expected_per_seed}"
            )

    return {
        "seeds": seeds,
        "baseline_fused_map@100": mean_std(
            [float(by_seed[seed]["baseline"]["fused"]["map@100"]) for seed in seeds]
        ),
        "baseline_anchor_map@100": mean_std(
            [float(by_seed[seed]["baseline"]["anchor"]["map@100"]) for seed in seeds]
        ),
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    checkpoint_hashes: set[str] = set()
    for path in args.result_json:
        item = json.loads(path.read_text(encoding="utf-8"))
        protocol = item.get("protocol", {})
        if protocol.get("test_materialized") is not False:
            raise ValueError(f"stress result is not validation-only: {path}")
        if protocol.get("selection_changes_permitted") is not False:
            raise ValueError(f"stress result permits post-freeze selection: {path}")
        dataset = str(item["dataset"])
        seed = int(item["seed"])
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate result for {dataset} seed {seed}")
        grouped[dataset][seed] = item
        checkpoint_hashes.add(str(item["checkpoint_sha256"]))

    dataset_summaries = {
        dataset: summarize_dataset(by_seed)
        for dataset, by_seed in sorted(grouped.items())
    }
    overall_guarantee = {
        cutoff: {
            "protected_anchor_hits": sum(
                int(condition["guarantee"]["by_cutoff"][cutoff]["protected_anchor_hits"])
                for dataset in dataset_summaries.values()
                for condition in dataset["conditions"].values()
            ),
            "violations": sum(
                int(condition["guarantee"]["by_cutoff"][cutoff]["violations"])
                for dataset in dataset_summaries.values()
                for condition in dataset["conditions"].values()
            ),
        }
        for cutoff in ("10", "50", "100")
    }
    summary = {
        "status": "postfreeze_validation_sensitivity_complete",
        "interpretation": (
            "input sensitivity only; not a future-world simulation, test result, "
            "or basis for model selection"
        ),
        "datasets": dataset_summaries,
        "stress_evaluation_conditions": sum(
            len(item["conditions"])
            for by_seed in grouped.values()
            for item in by_seed.values()
        ),
        "overall_guarantee": overall_guarantee,
        "checkpoint_sha256_count": len(checkpoint_hashes),
        "test_materialized": False,
        "selection_changes_permitted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for dataset, dataset_summary in summary["datasets"].items():
        print(dataset)
        for key, condition in dataset_summary["conditions"].items():
            delta = condition["fused_minus_unperturbed_map@100"]
            versus_anchor = condition["fused_minus_anchor_map@100"]
            print(
                f"  {key} input_delta={delta['mean']:.6f} "
                f"vs_anchor={versus_anchor['mean']:.6f} "
                f"violations={condition['guarantee']['violations']}"
            )


if __name__ == "__main__":
    main()
