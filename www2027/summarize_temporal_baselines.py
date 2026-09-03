"""Aggregate validation-only temporal baseline runs and audit their protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "n": len(values),
    }


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    protocol_errors: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for path in args.result_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload["model_name"], payload["dataset"], int(payload["seed"]))
        if key in seen:
            raise ValueError(f"duplicate run for {key}")
        seen.add(key)
        protocol = payload.get("protocol", {})
        for field in ("test_materialized", "test_evaluated", "test_used_for_selection"):
            if protocol.get(field) is not False:
                protocol_errors.append(f"{path}: {field} is not false")
        if protocol.get("train_graph_records") != "train_only":
            protocol_errors.append(f"{path}: graph is not train-only")
        grouped[(payload["dataset"], payload["model_name"])].append(payload)

    if protocol_errors:
        raise ValueError("protocol audit failed:\n" + "\n".join(protocol_errors))

    summary: dict[str, Any] = {
        "run_count": len(seen),
        "protocol_valid": True,
        "test_evaluated": False,
        "groups": {},
    }
    for (dataset, model), payloads in sorted(grouped.items()):
        group_key = f"{dataset}/{model}"
        metrics = payloads[0]["validation"].keys()
        summary["groups"][group_key] = {
            "seeds": sorted(int(payload["seed"]) for payload in payloads),
            "selected_epochs": [int(payload["selected_epoch"]) for payload in payloads],
            "parameter_count": int(payloads[0]["parameter_count"]),
            "metrics": {
                metric: summarize([float(payload["validation"][metric]) for payload in payloads])
                for metric in metrics
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
