"""Summarize paired multi-seed validation-only dual-path evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import re
import statistics


CUTOFFS = (10, 50, 100)
PROTECTED_STRATA = {
    "popularity": ("mid", "tail", "emerging"),
    "recency": ("historical_inactive", "emerging"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def exact_one_sided_sign_flip_p(deltas: list[float]) -> float:
    """Exact paired randomization p-value for a positive mean delta."""

    observed = statistics.fmean(deltas)
    tolerance = max(1.0, abs(observed)) * 1e-14
    permuted = [
        statistics.fmean(sign * delta for sign, delta in zip(signs, deltas))
        for signs in product((-1.0, 1.0), repeat=len(deltas))
    ]
    return sum(value >= observed - tolerance for value in permuted) / len(permuted)


def result_seed(result: dict[str, object]) -> int:
    if result.get("seed") is not None:
        return int(result["seed"])
    checkpoint_stem = Path(str(result["checkpoint"])).stem
    match = re.search(r"_s(\d+)$", checkpoint_stem)
    if match is None:
        raise ValueError(f"cannot recover seed from checkpoint {checkpoint_stem}")
    return int(match.group(1))


def load_results(paths: list[Path]) -> dict[str, dict[int, dict[str, object]]]:
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol", {}).get("test_materialized") is not False:
            raise ValueError(f"result is not validation-only: {path}")
        dataset = str(result["dataset"])
        seed = result_seed(result)
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate result for {dataset} seed {seed}")
        grouped[dataset][seed] = result
    return dict(grouped)


def summarize_dataset(
    by_seed: dict[int, dict[str, object]],
) -> dict[str, object]:
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
            anchor = [
                float(by_seed[seed]["anchor_validation"][key]) for seed in seeds
            ]
            fused = [
                float(by_seed[seed]["restored_validation"][key]) for seed in seeds
            ]
            deltas = [fused_value - anchor_value for fused_value, anchor_value in zip(fused, anchor)]
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
                    fused_stratum = item["validation_stratified"][taxonomy][name]
                    anchor_stratum = item["anchor_validation_stratified"][taxonomy][name]
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
    grouped = load_results(args.result_json)
    summary = {
        "status": "validation_only_test_locked",
        "method": "anchored_tpr_hierarchical_protected_union",
        "statistical_test": "exact_one_sided_paired_sign_flip_on_mean_delta",
        "datasets": {
            dataset: summarize_dataset(by_seed)
            for dataset, by_seed in sorted(grouped.items())
        },
        "test_materialized": False,
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
