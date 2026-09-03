"""Summarize post-freeze dynamic/static/no-mask anchor ablations."""

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


METHODS = ("dynamic", "static", "none")
METRICS = ("map", "worst_map", "hit", "worst_hit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-json", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def validate_protocol(path: Path, item: dict[str, object]) -> None:
    protocol = item.get("protocol", {})
    if protocol.get("test_materialized") is not False:
        raise ValueError(f"result is not validation-only: {path}")
    if protocol.get("selection_changes_permitted") is not False:
        raise ValueError(f"result permits model selection: {path}")


def load_dynamic(
    paths: list[Path],
) -> dict[str, dict[int, dict[str, object]]]:
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        validate_protocol(path, item)
        dataset, seed = str(item["dataset"]), int(item["seed"])
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate dynamic result for {dataset} seed {seed}")
        grouped[dataset][seed] = {
            "metrics": item["paths"]["anchor"]["metrics"],
            "stratified": item["paths"]["anchor"]["stratified"],
            "source": str(path),
        }
    return dict(grouped)


def load_ablations(
    paths: list[Path],
) -> dict[str, dict[str, dict[int, dict[str, object]]]]:
    grouped: defaultdict[str, defaultdict[str, dict[int, dict[str, object]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        validate_protocol(path, item)
        dataset = str(item["dataset"])
        method = str(item["mask_mode"])
        seed = int(item["seed"])
        if method not in ("static", "none"):
            raise ValueError(f"unexpected mask method {method}: {path}")
        if seed in grouped[dataset][method]:
            raise ValueError(f"duplicate {method} result for {dataset} seed {seed}")
        grouped[dataset][method][seed] = {
            "metrics": item["restored_validation"],
            "stratified": item["validation_stratified"],
            "selected_epoch": int(item["selected_epoch"]),
            "source": str(path),
        }
    return {
        dataset: {method: dict(by_seed) for method, by_seed in methods.items()}
        for dataset, methods in grouped.items()
    }


def comparison(dynamic: list[float], ablated: list[float]) -> dict[str, object]:
    deltas = [base - alternative for base, alternative in zip(dynamic, ablated)]
    return {
        "delta": mean_std(deltas),
        "positive_seeds": sum(delta > 0 for delta in deltas),
        "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(deltas),
        "per_seed_delta": deltas,
    }


def summarize_strata(
    methods: dict[str, dict[int, dict[str, object]]], seeds: list[int]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for taxonomy, names in PROTECTED_STRATA.items():
        result[taxonomy] = {}
        for name in names:
            group_summary: dict[str, object] = {}
            for cutoff in CUTOFFS:
                cutoff_summary: dict[str, object] = {}
                for metric_name in ("map", "hit"):
                    key = f"{metric_name}@{cutoff}"
                    values = {
                        method: [
                            float(
                                methods[method][seed]["stratified"][taxonomy][name][
                                    key
                                ]
                            )
                            for seed in seeds
                        ]
                        for method in METHODS
                    }
                    cutoff_summary[metric_name] = {
                        "methods": {
                            method: mean_std(method_values)
                            for method, method_values in values.items()
                        },
                        "dynamic_minus_static": comparison(
                            values["dynamic"], values["static"]
                        ),
                        "dynamic_minus_none": comparison(
                            values["dynamic"], values["none"]
                        ),
                    }
                group_summary[str(cutoff)] = cutoff_summary
            counts = {
                method: [
                    int(methods[method][seed]["stratified"][taxonomy][name]["count"])
                    for seed in seeds
                ]
                for method in METHODS
            }
            if not all(counts[method] == counts["dynamic"] for method in METHODS):
                raise ValueError(f"stratum counts differ for {taxonomy}/{name}")
            group_summary["count_per_seed"] = counts["dynamic"]
            result[taxonomy][name] = group_summary
    return result


def summarize_dataset(
    dynamic: dict[int, dict[str, object]],
    ablations: dict[str, dict[int, dict[str, object]]],
) -> dict[str, object]:
    methods = {"dynamic": dynamic, **ablations}
    if set(methods) != set(METHODS):
        raise ValueError(f"missing methods: expected {METHODS}, found {tuple(methods)}")
    seed_sets = {method: set(by_seed) for method, by_seed in methods.items()}
    if any(seed_set != seed_sets["dynamic"] for seed_set in seed_sets.values()):
        raise ValueError(f"seed mismatch: {seed_sets}")
    seeds = sorted(seed_sets["dynamic"])
    result: dict[str, object] = {
        "seeds": seeds,
        "selected_epoch": {
            method: mean_std(
                [float(methods[method][seed]["selected_epoch"]) for seed in seeds]
            )
            for method in ("static", "none")
        },
        "cutoffs": {},
    }
    for cutoff in CUTOFFS:
        cutoff_summary: dict[str, object] = {}
        for metric_name in METRICS:
            key = f"{metric_name}@{cutoff}"
            values = {
                method: [
                    float(methods[method][seed]["metrics"][key]) for seed in seeds
                ]
                for method in METHODS
            }
            cutoff_summary[metric_name] = {
                "methods": {
                    method: mean_std(method_values)
                    for method, method_values in values.items()
                },
                "dynamic_minus_static": comparison(
                    values["dynamic"], values["static"]
                ),
                "dynamic_minus_none": comparison(values["dynamic"], values["none"]),
                "per_seed": {
                    method: dict(zip(seeds, method_values))
                    for method, method_values in values.items()
                },
            }
        result["cutoffs"][str(cutoff)] = cutoff_summary
    result["protected_strata"] = summarize_strata(methods, seeds)
    return result


def main() -> None:
    args = parse_args()
    dynamic = load_dynamic(args.dynamic_json)
    ablations = load_ablations(args.mask_json)
    if set(dynamic) != set(ablations):
        raise ValueError(
            f"dataset mismatch: dynamic={set(dynamic)}, ablations={set(ablations)}"
        )
    summary = {
        "status": "postfreeze_validation_mask_ablation_complete",
        "comparison": "dynamic mask versus static mask and no mask at the anchor stage",
        "interpretation": "descriptive validation ablation; no post-test selection",
        "datasets": {
            dataset: summarize_dataset(dynamic[dataset], ablations[dataset])
            for dataset in sorted(dynamic)
        },
        "test_materialized": False,
        "selection_changes_permitted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for dataset, item in summary["datasets"].items():
        metric = item["cutoffs"]["100"]["map"]
        methods = metric["methods"]
        static = metric["dynamic_minus_static"]
        no_mask = metric["dynamic_minus_none"]
        print(
            f"{dataset} dynamic={methods['dynamic']['mean']:.6f} "
            f"static={methods['static']['mean']:.6f} "
            f"none={methods['none']['mean']:.6f} "
            f"dynamic-static={static['delta']['mean']:+.6f} "
            f"({static['positive_seeds']}/{len(item['seeds'])}, "
            f"p={static['exact_one_sided_sign_flip_p']:.5f}) "
            f"dynamic-none={no_mask['delta']['mean']:+.6f} "
            f"({no_mask['positive_seeds']}/{len(item['seeds'])}, "
            f"p={no_mask['exact_one_sided_sign_flip_p']:.5f})"
        )


if __name__ == "__main__":
    main()
