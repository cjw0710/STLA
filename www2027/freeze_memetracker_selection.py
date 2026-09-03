"""Freeze MemeTracker strong-adapter selection before opening its test file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .evaluate_checkpoint import file_sha256


ROOT = Path(__file__).resolve().parent
MODELS = ("dyhgcn", "disenidp")
SEEDS = (21, 42, 84, 126, 168)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "memetracker_pretest_selection_manifest.json",
    )
    return parser.parse_args()


def _load_validation_result(path: Path, *, status: str | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if status is not None and payload.get("status") != status:
        raise ValueError(f"unexpected result status in {path}")
    protocol = payload.get("protocol", {})
    if any(
        protocol.get(key) is not False
        for key in ("test_materialized", "test_evaluated", "test_used_for_selection")
    ):
        raise ValueError(f"test access was not sealed in {path}")
    return payload


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection manifest: {output}")

    dataset_dir = (args.dataset_root / "memetracker").resolve()
    split_manifest_path = dataset_dir / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    test_path = dataset_dir / "cascade_test.json"
    expected_test_hash = split_manifest["outputs"]["cascade_test.json"]
    if file_sha256(test_path) != expected_test_hash:
        raise ValueError("sealed MemeTracker test file differs from its split manifest")

    entries: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            stem = f"{model}_memetracker_seed{seed}"
            baseline_result_path = ROOT / "artifacts" / "postfreeze_temporal_baselines" / f"{stem}.json"
            adapter_result_path = ROOT / "artifacts" / "postfreeze_strong_adapter" / f"{stem}.json"
            adapter_checkpoint_path = ROOT / "checkpoints" / "postfreeze_strong_adapter" / f"{stem}.pt"
            baseline = _load_validation_result(
                baseline_result_path,
                status=None,
            )
            adapter = _load_validation_result(
                adapter_result_path,
                status="validation_only_strong_backbone_gate",
            )
            if baseline.get("dataset") != "memetracker" or adapter.get("dataset") != "memetracker":
                raise ValueError(f"dataset mismatch for {stem}")
            if int(baseline["seed"]) != seed or int(adapter["seed"]) != seed:
                raise ValueError(f"seed mismatch for {stem}")
            if str(baseline["model_name"]).lower() != model:
                raise ValueError(f"anchor model mismatch for {stem}")
            baseline_checkpoint_path = Path(str(baseline["checkpoint"])).resolve()
            entries.append(
                {
                    "anchor_model": baseline["model_name"],
                    "seed": seed,
                    "baseline_result": str(baseline_result_path.resolve()),
                    "baseline_result_sha256": file_sha256(baseline_result_path),
                    "baseline_checkpoint": str(baseline_checkpoint_path),
                    "baseline_checkpoint_sha256": file_sha256(baseline_checkpoint_path),
                    "baseline_selected_epoch": int(baseline["selected_epoch"]),
                    "adapter_result": str(adapter_result_path.resolve()),
                    "adapter_result_sha256": file_sha256(adapter_result_path),
                    "adapter_checkpoint": str(adapter_checkpoint_path.resolve()),
                    "adapter_checkpoint_sha256": file_sha256(adapter_checkpoint_path),
                    "adapter_selected_epoch": int(adapter["selected_epoch"]),
                }
            )

    evaluator = ROOT / "evaluate_memetracker_strong_test.py"
    manifest: dict[str, Any] = {
        "status": "frozen_before_memetracker_test",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "memetracker",
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "sealed_test": str(test_path),
        "sealed_test_sha256": expected_test_hash,
        "evaluator": str(evaluator),
        "evaluator_sha256": file_sha256(evaluator),
        "entries": entries,
        "test_configuration": {
            "test_environments": 3,
            "batch_size": 64,
            "max_prefix_length": 50,
            "sample_hop": 2,
            "protected_cutoffs": [10, 50, 100],
            "selection_changes_permitted": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"frozen_memetracker_selection entries={len(entries)} manifest={output}", flush=True)
    return manifest


def main() -> None:
    freeze(parse_args())


if __name__ == "__main__":
    main()
