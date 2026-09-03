"""Summarize paired multi-seed internal Dynamic DeDiff validation results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

from .summarize_validation_results import exact_one_sided_sign_flip_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        protocol = result.get("protocol", {})
        if result.get("status") != "validation_only_dynamic_internal_dediff":
            raise ValueError("unexpected dynamic DeDiff result status")
        if int(result.get("base_rank", 0)) != 0:
            raise ValueError("multi-seed summary accepts only uncompressed dynamic runs")
        if any(protocol.get(key) is not False for key in (
            "test_materialized", "test_evaluated", "test_used_for_selection"
        )):
            raise ValueError("result failed validation-only protocol audit")
        dataset = str(result["dataset"])
        seed = int(result["seed"])
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate {dataset} seed {seed}")
        grouped[dataset][seed] = result

    output: dict[str, Any] = {
        "status": "validation_only_dynamic_dediff_multiseed_summary",
        "datasets": {},
        "statistical_test": "exact_one_sided_paired_sign_flip_on_mean_delta",
        "protocol": {
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
        },
    }
    for dataset, by_seed in sorted(grouped.items()):
        seeds = sorted(by_seed)
        dataset_summary: dict[str, Any] = {
            "seeds": seeds,
            "selected_epoch_zero": sum(int(by_seed[seed]["selected_epoch"]) == 0 for seed in seeds),
            "metrics": {},
            "guarantee": {},
        }
        for metric in (
            "map@10", "map@50", "map@100", "worst_map@10", "worst_map@50", "worst_map@100",
            "hit@10", "hit@50", "hit@100", "worst_hit@10", "worst_hit@50", "worst_hit@100",
        ):
            anchor = [
                float(by_seed[seed]["evaluation"]["paths"]["anchor"]["metrics"][metric])
                for seed in seeds
            ]
            dynamic = [
                float(by_seed[seed]["evaluation"]["paths"]["hierarchical_union"]["metrics"][metric])
                for seed in seeds
            ]
            deltas = [right - left for left, right in zip(anchor, dynamic)]
            dataset_summary["metrics"][metric] = {
                "anchor": mean_std(anchor),
                "dynamic": mean_std(dynamic),
                "delta": mean_std(deltas),
                "paired_deltas": dict(zip(map(str, seeds), deltas)),
                "positive_seeds": sum(delta > 0 for delta in deltas),
                "negative_seeds": sum(delta < 0 for delta in deltas),
                "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(deltas),
            }
        for cutoff in (10, 50, 100):
            entries = [by_seed[seed]["evaluation"]["guarantee"][str(cutoff)] for seed in seeds]
            dataset_summary["guarantee"][str(cutoff)] = {
                "protected_anchor_hits": sum(int(entry["protected_anchor_hits"]) for entry in entries),
                "violations": sum(int(entry["violations"]) for entry in entries),
            }
        output["datasets"][dataset] = dataset_summary
    return output


def main() -> None:
    args = parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result_json]
    summary = summarize(results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for dataset, values in summary["datasets"].items():
        mean_map = values["metrics"]["map@100"]
        worst_map = values["metrics"]["worst_map@100"]
        print(
            f"{dataset}: delta={mean_map['delta']['mean']:+.6f} "
            f"({mean_map['positive_seeds']}/{len(values['seeds'])}, "
            f"p={mean_map['exact_one_sided_sign_flip_p']:.5f}); "
            f"worst={worst_map['delta']['mean']:+.6f} "
            f"p={worst_map['exact_one_sided_sign_flip_p']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

