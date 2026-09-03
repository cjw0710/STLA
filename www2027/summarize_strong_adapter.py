"""Summarize paired multi-seed validation results for strong logit adapters."""

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


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    grouped: defaultdict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in args.result_json:
        result = json.loads(path.read_text(encoding="utf-8"))
        protocol = result.get("protocol", {})
        if result.get("status") != "validation_only_strong_backbone_gate" or any(
            protocol.get(key) is not False
            for key in ("test_materialized", "test_evaluated", "test_used_for_selection")
        ):
            raise ValueError(f"invalid validation-only strong result: {path}")
        key = str(result["anchor_model"]), str(result["dataset"])
        seed = int(result["seed"])
        if seed in grouped[key]:
            raise ValueError(f"duplicate result for {key} seed {seed}")
        grouped[key][seed] = result

    output: dict[str, Any] = {
        "status": "validation_only_strong_adapter_multiseed_summary",
        "groups": {},
        "statistical_test": "exact_one_sided_paired_sign_flip_on_mean_delta",
        "protocol": {
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
        },
    }
    for (model, dataset), by_seed in sorted(grouped.items()):
        seeds = sorted(by_seed)
        item: dict[str, Any] = {"anchor_model": model, "dataset": dataset, "seeds": seeds, "metrics": {}}
        for metric in ("map@100", "worst_map@100"):
            anchor = [
                float(by_seed[seed]["evaluation"]["paths"]["anchor"]["metrics"][metric])
                for seed in seeds
            ]
            fused = [
                float(by_seed[seed]["evaluation"]["paths"]["hierarchical_union"]["metrics"][metric])
                for seed in seeds
            ]
            deltas = [right - left for left, right in zip(anchor, fused)]
            item["metrics"][metric] = {
                "anchor": stats(anchor),
                "fused": stats(fused),
                "delta": stats(deltas),
                "paired_deltas": dict(zip(map(str, seeds), deltas)),
                "positive_seeds": sum(delta > 0 for delta in deltas),
                "negative_seeds": sum(delta < 0 for delta in deltas),
                "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(deltas),
            }
        item["guarantee"] = {
            str(cutoff): {
                "protected_anchor_hits": sum(
                    int(by_seed[seed]["evaluation"]["guarantee"][str(cutoff)]["protected_anchor_hits"])
                    for seed in seeds
                ),
                "violations": sum(
                    int(by_seed[seed]["evaluation"]["guarantee"][str(cutoff)]["violations"])
                    for seed in seeds
                ),
            }
            for cutoff in (10, 50, 100)
        }
        output["groups"][f"{model}_{dataset}"] = item
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for key, item in output["groups"].items():
        metric = item["metrics"]["map@100"]
        print(
            f"{key}: delta={metric['delta']['mean']:+.6f}, "
            f"positive={metric['positive_seeds']}/{len(item['seeds'])}, "
            f"p={metric['exact_one_sided_sign_flip_p']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

