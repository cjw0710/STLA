"""Summarize the frozen MemeTracker paired one-shot test results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

from .summarize_validation_results import exact_one_sided_sign_flip_p


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: defaultdict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    manifest_hashes: set[str] = set()
    for path in args.result_json:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "locked_one_shot_memetracker_strong_adapter":
            raise ValueError(f"invalid one-shot result: {path}")
        protocol = result.get("protocol", {})
        if not protocol.get("test_evaluated") or protocol.get("test_used_for_selection") is not False:
            raise ValueError(f"invalid test protocol: {path}")
        model = str(result["anchor_model"])
        seed = int(result["seed"])
        if seed in grouped[model]:
            raise ValueError(f"duplicate result for {model} seed {seed}")
        grouped[model][seed] = result
        manifest_hashes.add(str(result["selection_manifest_sha256"]))
    if len(manifest_hashes) != 1:
        raise ValueError("test results do not share one frozen selection manifest")

    output: dict[str, Any] = {
        "status": "locked_one_shot_memetracker_summary",
        "selection_manifest_sha256": next(iter(manifest_hashes)),
        "groups": {},
        "statistical_test": "exact_one_sided_paired_sign_flip_on_test_delta",
    }
    for model, by_seed in sorted(grouped.items()):
        seeds = sorted(by_seed)
        item: dict[str, Any] = {"anchor_model": model, "seeds": seeds, "metrics": {}}
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
        output["groups"][model] = item

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for model, item in output["groups"].items():
        metric = item["metrics"]["map@100"]
        print(
            f"{model}: delta={metric['delta']['mean']:+.6f}, "
            f"positive={metric['positive_seeds']}/{len(item['seeds'])}, "
            f"p={metric['exact_one_sided_sign_flip_p']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
