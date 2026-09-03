"""Evaluate one frozen MemeTracker strong adapter exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .baselines.buzzbloom_temporal import TemporalBuzzLoader, _buzz_cascade_dataset, select_device
from .data import build_rolling_snapshots, load_cascades, make_temporal_environments
from .evaluate_checkpoint import file_sha256
from .models import TemporalLogitAdapter
from .train_strong_logit_adapter import (
    AdapterEnvironment,
    evaluate,
    load_frozen_anchor,
    make_loaders,
    seed_everything,
)
from .training import prepare_environment


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--adapter-result", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _matching_entry(
    manifest: dict[str, Any],
    baseline_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    matches = [
        entry
        for entry in manifest.get("entries", [])
        if Path(entry["baseline_result"]).resolve() == baseline_path
        and Path(entry["adapter_result"]).resolve() == adapter_path
    ]
    if len(matches) != 1:
        raise ValueError(f"selection manifest contains {len(matches)} matching entries")
    return matches[0]


def _build_test_environments(
    loader: TemporalBuzzLoader,
    test_records: Sequence[Any],
    *,
    test_count: int,
    max_prefix_length: int,
) -> list[AdapterEnvironment]:
    valid_groups = make_temporal_environments(loader.split.valid, 2, prefix="valid")
    test_groups = make_temporal_environments(test_records, test_count, prefix="test")
    warm_start = (*loader.split.train, *loader.split.valid)
    snapshots = build_rolling_snapshots(
        test_groups,
        loader.num_nodes,
        warm_start_records=warm_start,
        warm_start_recent_records=valid_groups[-1].records,
        sample_hop=2,
    )
    CascadeDataset = _buzz_cascade_dataset()
    environments: list[AdapterEnvironment] = []
    for snapshot in snapshots:
        records = snapshot.environment.records
        prepared = prepare_environment(snapshot, loader.num_nodes, max_prefix_length)
        environments.append(
            AdapterEnvironment(
                name=snapshot.environment.name,
                dataset=CascadeDataset(
                    [[node + 2 for node in record.cascade] for record in records],
                    [list(record.timestamp) for record in records],
                    [0] * len(records),
                ),
                environment_features=prepared.environment_features,
                historical_popularity=prepared.historical_popularity,
                recent_popularity=prepared.recent_popularity,
                popularity_groups=prepared.popularity_groups,
                recency_groups=prepared.recency_groups,
            )
        )
    return environments


def evaluate_once(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.selection_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_memetracker_test":
        raise ValueError("selection manifest is not frozen for MemeTracker test")
    evaluator_path = Path(__file__).resolve()
    if manifest.get("evaluator_sha256") != file_sha256(evaluator_path):
        raise ValueError("test evaluator differs from the frozen manifest")

    baseline_path = args.baseline_result.resolve()
    adapter_result_path = args.adapter_result.resolve()
    entry = _matching_entry(manifest, baseline_path, adapter_result_path)
    checks = (
        (baseline_path, "baseline_result_sha256"),
        (Path(entry["baseline_checkpoint"]), "baseline_checkpoint_sha256"),
        (adapter_result_path, "adapter_result_sha256"),
        (Path(entry["adapter_checkpoint"]), "adapter_checkpoint_sha256"),
    )
    for path, hash_key in checks:
        if file_sha256(path) != entry[hash_key]:
            raise ValueError(f"{path} differs from the frozen selection manifest")

    test_path = (args.dataset_root / "memetracker" / "cascade_test.json").resolve()
    if file_sha256(test_path) != manifest["sealed_test_sha256"]:
        raise ValueError("sealed test file differs from the frozen manifest")

    result_path = args.result_json.resolve()
    manifest_hash = file_sha256(manifest_path)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("selection_manifest_sha256") == manifest_hash
            and existing.get("baseline_checkpoint_sha256") == entry["baseline_checkpoint_sha256"]
            and existing.get("adapter_checkpoint_sha256") == entry["adapter_checkpoint_sha256"]
        ):
            print(f"memetracker_test_already_evaluated result={result_path}", flush=True)
            return existing
        raise FileExistsError(f"refusing to overwrite one-shot test result: {result_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    adapter_result = json.loads(adapter_result_path.read_text(encoding="utf-8"))
    seed = int(entry["seed"])
    if int(baseline["selected_epoch"]) != int(entry["baseline_selected_epoch"]):
        raise ValueError("baseline selected epoch differs from frozen entry")
    if int(adapter_result["selected_epoch"]) != int(entry["adapter_selected_epoch"]):
        raise ValueError("adapter selected epoch differs from frozen entry")

    seed_everything(seed)
    device = select_device(args.device)
    config = manifest["test_configuration"]
    max_prefix_length = int(config["max_prefix_length"])
    loader = TemporalBuzzLoader(
        "memetracker",
        args.dataset_root,
        max_prefix_length=max_prefix_length,
        valid_environments=2,
    )
    if not loader.strict_prepartitioned or loader.test_materialized:
        raise RuntimeError("MemeTracker selection loader is not sealed")
    anchor, patches = load_frozen_anchor(baseline, loader, device)
    adapter = TemporalLogitAdapter(
        loader.num_nodes,
        context_dim=16,
        hidden_dim=64,
        environment_hidden_dim=32,
        dropout=0.2,
    ).to(device)
    adapter_checkpoint = torch.load(
        Path(entry["adapter_checkpoint"]),
        map_location=device,
        weights_only=False,
    )
    if int(adapter_checkpoint["selected_epoch"]) != int(entry["adapter_selected_epoch"]):
        raise ValueError("adapter checkpoint epoch differs from frozen entry")
    adapter.load_state_dict(adapter_checkpoint["adapter_state"])

    # This is the first operation that opens the sealed test payload.  Every
    # manifest, hash, checkpoint, and output-path guard above has already passed.
    test_records = load_cascades(args.dataset_root / "memetracker", split_names=("test",))
    if len(test_records) != loader.test_record_count:
        raise ValueError("test record count differs from the sealed split manifest")
    test_environments = [
        environment.context_to(device)
        for environment in _build_test_environments(
            loader,
            test_records,
            test_count=int(config["test_environments"]),
            max_prefix_length=max_prefix_length,
        )
    ]
    test_loaders = make_loaders(
        test_environments,
        batch_size=int(config["batch_size"]),
        max_prefix_length=max_prefix_length,
        shuffle=False,
        seed=seed,
    )
    evaluation = evaluate(
        anchor,
        adapter,
        test_environments,
        test_loaders,
        device,
        max_batches=0,
    )
    result: dict[str, Any] = {
        "status": "locked_one_shot_memetracker_strong_adapter",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "memetracker",
        "seed": seed,
        "anchor_model": baseline["model_name"],
        "baseline_checkpoint": entry["baseline_checkpoint"],
        "baseline_checkpoint_sha256": entry["baseline_checkpoint_sha256"],
        "adapter_checkpoint": entry["adapter_checkpoint"],
        "adapter_checkpoint_sha256": entry["adapter_checkpoint_sha256"],
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": manifest_hash,
        "semantics_preserving_anchor_patches": patches,
        "evaluation": evaluation,
        "protocol": {
            "chronological_split": [0.7, 0.1, 0.2],
            "timestamp_ties_preserved": True,
            "test_environments": int(config["test_environments"]),
            "batch_size": int(config["batch_size"]),
            "protected_cutoffs": config["protected_cutoffs"],
            "test_materialized": True,
            "test_evaluated": True,
            "test_used_for_selection": False,
            "selection_changes_permitted": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = evaluation["paths"]["hierarchical_union"]["metrics"]
    print(
        f"one_shot_memetracker model={baseline['model_name']} seed={seed} "
        f"map@100={metrics['map@100']:.6f} worst={metrics['worst_map@100']:.6f}",
        flush=True,
    )
    return result


def main() -> None:
    evaluate_once(parse_args())


if __name__ == "__main__":
    main()
