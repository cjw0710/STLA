"""Run paired multi-seed screening experiments for WWW DriftDiff."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import sys


METHODS = {
    "static_erm": {"mask_mode": "static", "objective": "erm"},
    "dynamic_erm": {"mask_mode": "dynamic", "objective": "erm"},
    "dynamic_temporal_prior": {
        "mask_mode": "dynamic",
        "prior_mode": "temporal",
        "objective": "erm",
    },
    "dynamic_anchored_tpr": {
        "mask_mode": "dynamic",
        "prior_mode": "temporal",
        "objective": "erm",
        "initialize_from": "dynamic_erm",
        "freeze_backbone": True,
        "preservation_weight": 1.0,
        "preservation_topk": 100,
        "preservation_margin": 0.0,
    },
    "static_balanced": {
        "mask_mode": "static",
        "objective": "erm",
        "popularity_balance_alpha": 0.25,
        "dormant_boost": 0.5,
    },
    "dynamic_balanced": {
        "mask_mode": "dynamic",
        "objective": "erm",
        "popularity_balance_alpha": 0.25,
        "dormant_boost": 0.5,
    },
    "dynamic_constrained_005": {
        "mask_mode": "dynamic",
        "objective": "erm",
        "constraint_weight": 0.05,
        "constraint_margin": 0.5,
    },
    "dynamic_constrained_010": {
        "mask_mode": "dynamic",
        "objective": "erm",
        "constraint_weight": 0.1,
        "constraint_margin": 0.5,
    },
    "dynamic_constrained_020": {
        "mask_mode": "dynamic",
        "objective": "erm",
        "constraint_weight": 0.2,
        "constraint_margin": 0.5,
    },
    "dynamic_groupdro": {"mask_mode": "dynamic", "objective": "groupdro"},
    "dynamic_vrex": {"mask_mode": "dynamic", "objective": "vrex"},
    "no_mask_erm": {"mask_mode": "none", "objective": "erm"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["christian", "android"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[21, 42, 84])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=["static_erm", "dynamic_erm"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prefix-length", type=int, default=50)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--environment-hidden-dim", type=int, default=16)
    parser.add_argument("--vrex-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "pilot",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "pilot",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    seed: int,
) -> Path:
    configuration = METHODS[method]
    run_name = f"pilot_{method}_s{seed}"
    result_path = args.output_dir / f"{dataset}_{method}_s{seed}.json"
    if result_path.exists() and not args.force:
        print(f"skip existing {result_path.name}", flush=True)
        return result_path

    command = [
        sys.executable,
        "-m",
        "www2027.train_temporal",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--mask-mode",
        configuration["mask_mode"],
        "--prior-mode",
        configuration.get("prior_mode", "none"),
        "--objective",
        configuration["objective"],
        "--run-name",
        run_name,
        "--method-label",
        method,
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--minimum-epochs",
        str(args.minimum_epochs),
        "--minimum-delta",
        str(args.minimum_delta),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--batch-size",
        str(args.batch_size),
        "--max-prefix-length",
        str(args.max_prefix_length),
        "--dimension",
        str(args.dimension),
        "--rank",
        str(args.rank),
        "--context-dim",
        str(args.context_dim),
        "--environment-hidden-dim",
        str(args.environment_hidden_dim),
        "--vrex-weight",
        str(args.vrex_weight),
        "--popularity-balance-alpha",
        str(configuration.get("popularity_balance_alpha", 0.0)),
        "--dormant-boost",
        str(configuration.get("dormant_boost", 0.0)),
        "--constraint-weight",
        str(configuration.get("constraint_weight", 0.0)),
        "--constraint-margin",
        str(configuration.get("constraint_margin", 0.5)),
        "--preservation-weight",
        str(configuration.get("preservation_weight", 0.0)),
        "--preservation-topk",
        str(configuration.get("preservation_topk", 100)),
        "--preservation-margin",
        str(configuration.get("preservation_margin", 0.0)),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--result-json",
        str(result_path),
    ]
    if configuration.get("initialize_from"):
        base_method = str(configuration["initialize_from"])
        initialization_path = (
            args.checkpoint_dir
            / f"{dataset}_pilot_{base_method}_s{seed}.pt"
        )
        if not initialization_path.exists():
            raise FileNotFoundError(
                f"{method} requires the paired base checkpoint {initialization_path}"
            )
        command.extend(["--initialize-checkpoint", str(initialization_path)])
    if configuration.get("freeze_backbone"):
        command.append("--freeze-backbone")
    if args.skip_test:
        command.append("--skip-test")
    print(f"\nrun dataset={dataset} method={method} seed={seed}", flush=True)
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    return result_path


def mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(
    args: argparse.Namespace,
    result_paths: list[Path],
) -> dict[str, object]:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    evaluation_key = "restored_validation" if args.skip_test else "test"
    stratified_key = "validation_stratified" if args.skip_test else "test_stratified"
    metrics = ("map@100", "worst_map@100", "hit@100", "worst_hit@100")
    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_run: dict[tuple[str, str, int], dict[str, object]] = {}
    for result in results:
        key = (
            str(result["dataset"]),
            str(result.get("method_label") or f"{result['mask_mode']}_{result['objective']}"),
        )
        grouped[key].append(result)
        by_run[(key[0], key[1], int(result["seed"]))] = result

    summary_rows: list[dict[str, object]] = []
    for (dataset, method), runs in sorted(grouped.items()):
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "seeds": sorted(int(run["seed"]) for run in runs),
        }
        for metric in metrics:
            row[metric] = mean_and_std(
                [float(run[evaluation_key][metric]) for run in runs]
            )
        row["stratified"] = {}
        for taxonomy, groups in runs[0][stratified_key].items():
            row["stratified"][taxonomy] = {}
            for group_name in groups:
                available = [
                    run[stratified_key][taxonomy][group_name]
                    for run in runs
                    if run[stratified_key][taxonomy][group_name].get("count", 0)
                ]
                group_summary: dict[str, object] = {
                    "count": sum(int(group["count"]) for group in available)
                }
                if available:
                    for metric in ("map@100", "hit@100"):
                        group_summary[metric] = mean_and_std(
                            [float(group[metric]) for group in available]
                        )
                row["stratified"][taxonomy][group_name] = group_summary
        summary_rows.append(row)

    paired_deltas: list[dict[str, object]] = []
    for dataset in args.datasets:
        for candidate in args.methods:
            if candidate == "static_erm":
                continue
            baseline = (
                "static_balanced"
                if candidate == "dynamic_balanced" and "static_balanced" in args.methods
                else "dynamic_erm"
                if (
                    candidate.startswith("dynamic_constrained")
                    or candidate in {
                        "dynamic_temporal_prior",
                        "dynamic_anchored_tpr",
                    }
                )
                and "dynamic_erm" in args.methods
                else "static_erm"
            )
            if candidate == baseline:
                continue
            available_seeds = [
                seed
                for seed in args.seeds
                if (dataset, baseline, seed) in by_run
                and (dataset, candidate, seed) in by_run
            ]
            if not available_seeds:
                continue
            row = {
                "dataset": dataset,
                "candidate": candidate,
                "baseline": baseline,
                "seeds": available_seeds,
            }
            for metric in metrics:
                differences = [
                    float(by_run[(dataset, candidate, seed)][evaluation_key][metric])
                    - float(by_run[(dataset, baseline, seed)][evaluation_key][metric])
                    for seed in available_seeds
                ]
                row[f"delta_{metric}"] = mean_and_std(differences)
            paired_deltas.append(row)

    return {
        "status": "screening_only_not_paper_results",
        "evaluation_split": "validation" if args.skip_test else "test",
        "configuration": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "methods": args.methods,
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "minimum_epochs": args.minimum_epochs,
            "skip_test": args.skip_test,
            "steps_per_epoch": args.steps_per_epoch,
            "max_eval_batches": args.max_eval_batches,
            "batch_size": args.batch_size,
            "max_prefix_length": args.max_prefix_length,
            "dimension": args.dimension,
            "rank": args.rank,
            "vrex_weight": args.vrex_weight,
        },
        "summary": summary_rows,
        "paired_comparisons": paired_deltas,
    }


def print_summary(summary: dict[str, object]) -> None:
    print(
        f"\nScreening summary on {summary['evaluation_split']} "
        "(mean +/- sample std)"
    )
    print(f"{'dataset':<12} {'method':<20} {'MAP@100':>18} {'worst MAP@100':>18}")
    print("-" * 72)
    for row in summary["summary"]:
        map_metric = row["map@100"]
        worst_metric = row["worst_map@100"]
        print(
            f"{row['dataset']:<12} {row['method']:<20} "
            f"{map_metric['mean']:.4f} +/- {map_metric['std']:.4f} "
            f"{worst_metric['mean']:.4f} +/- {worst_metric['std']:.4f}"
        )
    print("\nPaired deltas")
    for row in summary["paired_comparisons"]:
        delta = row["delta_map@100"]
        worst_delta = row["delta_worst_map@100"]
        print(
            f"{row['dataset']} {row['candidate']} - {row['baseline']}: "
            f"MAP@100 {delta['mean']:+.4f}, "
            f"worst MAP@100 {worst_delta['mean']:+.4f}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        run_one(args, dataset, method, seed)
        for dataset in args.datasets
        for method in args.methods
        for seed in args.seeds
    ]
    summary = aggregate(args, paths)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
