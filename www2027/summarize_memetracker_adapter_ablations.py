"""Summarize post-confirmation MemeTracker validation-only adapter ablations."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

from .summarize_validation_results import exact_one_sided_sign_flip_p


ROOT = Path(__file__).resolve().parent
MODELS = ("dyhgcn", "disenidp")
ABLATIONS = ("no_environment", "no_prefix", "historical_only")
SEEDS = (21, 42, 84, 126, 168)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def read_result(path: Path, expected_ablation: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    actual_ablation = payload.get("adapter_ablation", "full")
    if (
        payload.get("status") != "validation_only_strong_backbone_gate"
        or actual_ablation != expected_ablation
        or any(
            protocol.get(key) is not False
            for key in ("test_materialized", "test_evaluated", "test_used_for_selection")
        )
    ):
        raise ValueError(f"invalid validation-only result: {path}")
    if expected_ablation != "full" and protocol.get("confirmatory_test_reused") is not False:
        raise ValueError(f"ablation does not certify zero confirmatory-test reuse: {path}")
    return payload


def main() -> None:
    full_root = ROOT / "artifacts" / "postfreeze_strong_adapter"
    ablation_root = ROOT / "artifacts" / "posttest_validation_ablation"
    output_path = ROOT / "artifacts" / "posttest_validation_ablation_summary.json"
    output: dict[str, Any] = {
        "status": "post_confirmation_validation_only_component_ablation",
        "models": {},
        "protocol": {
            "dataset": "memetracker",
            "partition": "validation",
            "test_materialized": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "confirmatory_test_reused": False,
            "interpretation": "post-confirmation descriptive analysis only",
        },
    }
    for model in MODELS:
        by_configuration: dict[str, list[dict[str, Any]]] = {}
        by_configuration["full"] = [
            read_result(
                full_root / f"{model}_memetracker_seed{seed}.json",
                "full",
            )
            for seed in SEEDS
        ]
        for ablation in ABLATIONS:
            by_configuration[ablation] = [
                read_result(
                    ablation_root / f"{model}_memetracker_seed{seed}_{ablation}.json",
                    ablation,
                )
                for seed in SEEDS
            ]

        anchor = [
            float(result["evaluation"]["paths"]["anchor"]["metrics"]["map@100"])
            for result in by_configuration["full"]
        ]
        full = [
            float(result["evaluation"]["paths"]["hierarchical_union"]["metrics"]["map@100"])
            for result in by_configuration["full"]
        ]
        model_item: dict[str, Any] = {
            "seeds": list(SEEDS),
            "anchor_map@100": stats(anchor),
            "configurations": {},
        }
        for configuration, results in by_configuration.items():
            values = [
                float(result["evaluation"]["paths"]["hierarchical_union"]["metrics"]["map@100"])
                for result in results
            ]
            worst_values = [
                float(result["evaluation"]["paths"]["hierarchical_union"]["metrics"]["worst_map@100"])
                for result in results
            ]
            gains = [value - base for value, base in zip(values, anchor)]
            full_minus_configuration = [reference - value for reference, value in zip(full, values)]
            model_item["configurations"][configuration] = {
                "map@100": stats(values),
                "worst_map@100": stats(worst_values),
                "gain_over_anchor": stats(gains),
                "positive_over_anchor_seeds": sum(value > 0 for value in gains),
                "full_minus_configuration": stats(full_minus_configuration),
                "full_better_seeds": sum(value > 0 for value in full_minus_configuration),
                "configuration_better_seeds": sum(value < 0 for value in full_minus_configuration),
                "one_sided_p_full_better": exact_one_sided_sign_flip_p(full_minus_configuration),
                "selected_epoch_zero_seeds": sum(int(result["selected_epoch"]) == 0 for result in results),
                "guarantee_violations": sum(
                    int(result["evaluation"]["guarantee"][str(cutoff)]["violations"])
                    for result in results
                    for cutoff in (10, 50, 100)
                ),
            }
        output["models"][model] = model_item

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
