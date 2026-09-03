"""Summarize frozen validation-only inference-chain ablations."""

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


PATHS = ("anchor", "adaptive", "top100_union", "hierarchical_union")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def summarize_dataset(by_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    seeds = sorted(by_seed)
    result: dict[str, object] = {"seeds": seeds, "paths": {}, "guarantee": {}}
    for path in PATHS:
        path_summary: dict[str, object] = {}
        for cutoff in CUTOFFS:
            metrics: dict[str, object] = {}
            for metric_name in ("map", "worst_map", "hit", "worst_hit"):
                key = f"{metric_name}@{cutoff}"
                values = [
                    float(by_seed[seed]["paths"][path]["metrics"][key])
                    for seed in seeds
                ]
                anchor = [
                    float(by_seed[seed]["paths"]["anchor"]["metrics"][key])
                    for seed in seeds
                ]
                deltas = [value - base for value, base in zip(values, anchor)]
                metrics[metric_name] = {
                    "value": mean_std(values),
                    "delta_vs_anchor": mean_std(deltas),
                    "positive_seeds": sum(delta > 0 for delta in deltas),
                    "exact_one_sided_sign_flip_p": (
                        exact_one_sided_sign_flip_p(deltas)
                        if path != "anchor"
                        else None
                    ),
                }
            path_summary[str(cutoff)] = metrics
        result["paths"][path] = path_summary

    for path in ("top100_union", "hierarchical_union"):
        result["guarantee"][path] = {}
        for cutoff in CUTOFFS:
            items = [
                by_seed[seed]["guarantee"][path][str(cutoff)] for seed in seeds
            ]
            protected_deltas: list[float] = []
            hit_key = f"hit@{cutoff}"
            for seed in seeds:
                for taxonomy, names in PROTECTED_STRATA.items():
                    for name in names:
                        path_stratum = by_seed[seed]["paths"][path]["stratified"][
                            taxonomy
                        ][name]
                        anchor_stratum = by_seed[seed]["paths"]["anchor"][
                            "stratified"
                        ][taxonomy][name]
                        if int(path_stratum["count"]):
                            protected_deltas.append(
                                float(path_stratum[hit_key])
                                - float(anchor_stratum[hit_key])
                            )
            result["guarantee"][path][str(cutoff)] = {
                "protected_anchor_hits": sum(
                    int(item["protected_anchor_hits"]) for item in items
                ),
                "violations": sum(int(item["violations"]) for item in items),
                "protected_stratum_minimum_hit_delta": (
                    min(protected_deltas) if protected_deltas else None
                ),
            }
    return result


def main() -> None:
    args = parse_args()
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in args.result_json:
        item = json.loads(path.read_text(encoding="utf-8"))
        protocol = item.get("protocol", {})
        if protocol.get("test_materialized") is not False:
            raise ValueError(f"ablation is not validation-only: {path}")
        if protocol.get("selection_changes_permitted") is not False:
            raise ValueError(f"ablation permits model selection: {path}")
        dataset, seed = str(item["dataset"]), int(item["seed"])
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate result for {dataset} seed {seed}")
        grouped[dataset][seed] = item

    summary = {
        "status": "postfreeze_validation_inference_ablation_complete",
        "interpretation": "descriptive validation ablation; no post-test selection",
        "datasets": {
            dataset: summarize_dataset(by_seed)
            for dataset, by_seed in sorted(grouped.items())
        },
        "test_materialized": False,
        "selection_changes_permitted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for dataset, dataset_summary in summary["datasets"].items():
        print(dataset)
        for path in PATHS:
            item = dataset_summary["paths"][path]["100"]["map"]
            print(
                f"  {path} map@100={item['value']['mean']:.6f} "
                f"delta={item['delta_vs_anchor']['mean']:.6f}"
            )


if __name__ == "__main__":
    main()
