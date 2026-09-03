"""Compare frozen final checkpoints with no-preservation-loss retraining."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .summarize_validation_results import (
    CUTOFFS,
    exact_one_sided_sign_flip_p,
    mean_std,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-json", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--no-preservation-json", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def result_seed(path: Path, item: dict[str, object]) -> int:
    if item.get("seed") is not None:
        return int(item["seed"])
    match = re.search(r"_s(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot recover seed from {path}")
    return int(match.group(1))


def load(paths: list[Path]) -> dict[str, dict[int, dict[str, object]]]:
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol", {}).get("test_materialized") is not False:
            raise ValueError(f"result is not validation-only: {path}")
        dataset, seed = str(item["dataset"]), result_seed(path, item)
        if seed in grouped[dataset]:
            raise ValueError(f"duplicate {dataset} seed {seed}")
        grouped[dataset][seed] = item
    return dict(grouped)


def main_metrics(item: dict[str, object], path: str) -> dict[str, float]:
    return item["paths"][path]["metrics"]


def summarize_dataset(
    main: dict[int, dict[str, object]],
    no_preservation: dict[int, dict[str, object]],
) -> dict[str, object]:
    seeds = sorted(set(main) & set(no_preservation))
    if seeds != sorted(main) or seeds != sorted(no_preservation):
        raise ValueError("main and no-preservation seeds differ")
    result: dict[str, object] = {
        "seeds": seeds,
        "selected_epoch": {
            "main": mean_std([float(main[seed]["selected_epoch"]) for seed in seeds]),
            "no_preservation": mean_std(
                [float(no_preservation[seed]["selected_epoch"]) for seed in seeds]
            ),
        },
        "cutoffs": {},
        "guarantee": {},
    }
    for cutoff in CUTOFFS:
        cutoff_result: dict[str, object] = {}
        for metric_name in ("map", "worst_map", "hit", "worst_hit"):
            key = f"{metric_name}@{cutoff}"
            main_values = [
                float(main_metrics(main[seed], "hierarchical_union")[key])
                for seed in seeds
            ]
            no_preservation_values = [
                float(no_preservation[seed]["restored_validation"][key])
                for seed in seeds
            ]
            deltas = [
                final - ablated
                for final, ablated in zip(main_values, no_preservation_values)
            ]
            cutoff_result[metric_name] = {
                "main": mean_std(main_values),
                "no_preservation": mean_std(no_preservation_values),
                "main_minus_no_preservation": mean_std(deltas),
                "positive_seeds": sum(delta > 0 for delta in deltas),
                "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(deltas),
            }

        main_fusion_cost = [
            float(main_metrics(main[seed], "hierarchical_union")[f"map@{cutoff}"])
            - float(main_metrics(main[seed], "adaptive")[f"map@{cutoff}"])
            for seed in seeds
        ]
        no_preservation_fusion_cost = [
            float(no_preservation[seed]["restored_validation"][f"map@{cutoff}"])
            - float(no_preservation[seed]["adaptive_validation"][f"map@{cutoff}"])
            for seed in seeds
        ]
        cutoff_result["hierarchical_minus_adaptive_map"] = {
            "main": mean_std(main_fusion_cost),
            "no_preservation": mean_std(no_preservation_fusion_cost),
        }
        result["cutoffs"][str(cutoff)] = cutoff_result

        main_guarantee = [
            main[seed]["guarantee"]["hierarchical_union"][str(cutoff)]
            for seed in seeds
        ]
        no_preservation_guarantee = [
            no_preservation[seed]["guarantee"]["by_cutoff"][str(cutoff)]
            for seed in seeds
        ]
        result["guarantee"][str(cutoff)] = {
            "main_violations": sum(int(item["violations"]) for item in main_guarantee),
            "no_preservation_violations": sum(
                int(item["violations"]) for item in no_preservation_guarantee
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    main_results = load(args.main_json)
    no_preservation_results = load(args.no_preservation_json)
    if set(main_results) != set(no_preservation_results):
        raise ValueError("main and no-preservation datasets differ")
    summary = {
        "status": "postfreeze_validation_preservation_ablation_complete",
        "comparison": "main preservation_weight=1 minus preservation_weight=0",
        "datasets": {
            dataset: summarize_dataset(
                main_results[dataset], no_preservation_results[dataset]
            )
            for dataset in sorted(main_results)
        },
        "test_materialized": False,
        "selection_changes_permitted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for dataset, item in summary["datasets"].items():
        metric = item["cutoffs"]["100"]["map"]
        fusion = item["cutoffs"]["100"]["hierarchical_minus_adaptive_map"]
        print(
            f"{dataset} main_minus_no_pres={metric['main_minus_no_preservation']['mean']:+.6f} "
            f"positive={metric['positive_seeds']}/5 "
            f"p={metric['exact_one_sided_sign_flip_p']:.5f} "
            f"fusion_main={fusion['main']['mean']:+.6f} "
            f"fusion_no_pres={fusion['no_preservation']['mean']:+.6f}"
        )


if __name__ == "__main__":
    main()
