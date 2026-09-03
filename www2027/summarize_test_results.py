"""Summarize frozen one-shot hierarchical test evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .summarize_validation_results import (
    CUTOFFS,
    PROTECTED_STRATA,
    exact_one_sided_sign_flip_p,
    mean_std,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def summarize_dataset(by_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    seeds = sorted(by_seed)
    result: dict[str, object] = {
        "seeds": seeds,
        "cutoffs": {},
        "guarantee": {},
        "protected_stratum_minimum_hit_delta": {},
    }
    for cutoff in CUTOFFS:
        cutoff_result: dict[str, object] = {}
        for metric_name in ("map", "worst_map", "hit", "worst_hit"):
            key = f"{metric_name}@{cutoff}"
            anchor = [float(by_seed[seed]["anchor_test"][key]) for seed in seeds]
            fused = [float(by_seed[seed]["test"][key]) for seed in seeds]
            deltas = [
                fused_value - anchor_value
                for fused_value, anchor_value in zip(fused, anchor)
            ]
            cutoff_result[metric_name] = {
                "anchor": mean_std(anchor),
                "fused": mean_std(fused),
                "delta": mean_std(deltas),
                "paired_deltas": dict(zip(map(str, seeds), deltas)),
                "positive_seeds": sum(delta > 0 for delta in deltas),
                "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(deltas),
            }
        result["cutoffs"][str(cutoff)] = cutoff_result

        guarantee = [
            by_seed[seed]["guarantee"]["by_cutoff"][str(cutoff)]
            for seed in seeds
        ]
        result["guarantee"][str(cutoff)] = {
            "protected_anchor_hits": sum(
                int(item["protected_anchor_hits"]) for item in guarantee
            ),
            "violations": sum(int(item["violations"]) for item in guarantee),
        }

        protected_deltas: list[float] = []
        key = f"hit@{cutoff}"
        for seed in seeds:
            item = by_seed[seed]
            for taxonomy, names in PROTECTED_STRATA.items():
                for name in names:
                    fused_stratum = item["test_stratified"][taxonomy][name]
                    anchor_stratum = item["anchor_test_stratified"][taxonomy][name]
                    if int(fused_stratum["count"]):
                        protected_deltas.append(
                            float(fused_stratum[key]) - float(anchor_stratum[key])
                        )
        result["protected_stratum_minimum_hit_delta"][str(cutoff)] = (
            min(protected_deltas) if protected_deltas else None
        )
    return result


def main() -> None:
    args = parse_args()
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    manifest_hashes: set[str] = set()
    for path in args.result_json:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol", {}).get("test_materialized") is not True:
            raise ValueError(f"result is not a one-shot test result: {path}")
        dataset = str(item["dataset"])
        seed = int(item["seed"])
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate result for {dataset} seed {seed}")
        grouped[dataset][seed] = item
        manifest_hashes.add(str(item["selection_manifest_sha256"]))
    if len(manifest_hashes) != 1:
        raise ValueError("test results do not share one frozen selection manifest")

    summary = {
        "status": "frozen_one_shot_test_complete",
        "method": "anchored_tpr_hierarchical_protected_union",
        "selection_manifest_sha256": next(iter(manifest_hashes)),
        "statistical_test": "exact_one_sided_paired_sign_flip_on_mean_delta",
        "datasets": {
            dataset: summarize_dataset(by_seed)
            for dataset, by_seed in sorted(grouped.items())
        },
        "test_materialized": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for dataset, dataset_summary in summary["datasets"].items():
        map_summary = dataset_summary["cutoffs"]["100"]["map"]
        worst_summary = dataset_summary["cutoffs"]["100"]["worst_map"]
        print(
            f"{dataset} delta_map@100={map_summary['delta']['mean']:.6f} "
            f"p={map_summary['exact_one_sided_sign_flip_p']:.5f} "
            f"delta_worst={worst_summary['delta']['mean']:.6f} "
            f"p={worst_summary['exact_one_sided_sign_flip_p']:.5f}"
        )


if __name__ == "__main__":
    main()
