"""Prepare the untouched BuzzBloom MemeTracker data for temporal confirmation.

The released timestamps omit a decimal point after the first six digits.  For
example, ``3383039575`` denotes ``338303.9575``.  Restoring that delimiter
makes every released cascade temporally nondecreasing.  This script performs
only deterministic parsing, chronological splitting, and ID remapping; it does
not compute targets, metrics, or model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

from .data.temporal_split import CascadeRecord, chronological_split


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "third_party" / "buzzbloom" / "data" / "memetracker"
DEFAULT_OUTPUT = ROOT.parent / "dataset" / "memetracker"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_compact_timestamp(raw: str) -> float:
    """Restore the omitted decimal delimiter in a released timestamp."""

    value = raw.strip()
    if not value:
        raise ValueError("timestamp cannot be empty")
    if "." in value:
        return float(value)
    if not value.isdigit():
        raise ValueError(f"timestamp must contain digits only: {raw!r}")
    if len(value) <= 6:
        return float(value)
    return float(f"{value[:6]}.{value[6:]}")


def load_user_mapping(source: Path) -> dict[str, int]:
    with (source / "u2idx.pickle").open("rb") as stream:
        mapping = pickle.load(stream)
    if not isinstance(mapping, dict):
        raise TypeError("u2idx.pickle must contain a dictionary")
    if mapping.get("<blank>") != 0 or mapping.get("</s>") != 1:
        raise ValueError("unexpected special-token mapping")
    return {str(key): int(value) for key, value in mapping.items()}


def load_records(source: Path, mapping: dict[str, int]) -> tuple[list[CascadeRecord], dict[str, int]]:
    records: list[CascadeRecord] = []
    filtered_too_long = 0
    with (source / "cascades.txt").open("r", encoding="utf-8") as stream:
        for source_index, line in enumerate(stream):
            if not line.strip():
                continue
            cascade: list[int] = []
            timestamps: list[float] = []
            for chunk in line.strip().split(","):
                parts = chunk.strip().split()
                if len(parts) != 2:
                    raise ValueError(f"malformed cascade chunk at line {source_index + 1}: {chunk!r}")
                user, raw_timestamp = parts
                if user not in mapping:
                    raise KeyError(f"unmapped user {user!r} at line {source_index + 1}")
                cascade.append(mapping[user] - 2)
                timestamps.append(parse_compact_timestamp(raw_timestamp))
            if len(cascade) > 500:
                filtered_too_long += 1
                continue
            if len(cascade) < 5:
                continue
            records.append(
                CascadeRecord(
                    cascade=tuple(cascade),
                    timestamp=tuple(timestamps),
                    source_split="raw",
                    source_index=source_index,
                )
            )
    return records, {"filtered_too_long": filtered_too_long}


def remap_edges(source: Path, mapping: dict[str, int]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with (source / "edges.txt").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.strip().split(",")]
            if len(parts) != 2 or any(item not in mapping for item in parts):
                raise ValueError(f"malformed or unmapped edge at line {line_number}")
            edge = (mapping[parts[0]] - 2, mapping[parts[1]] - 2)
            if edge[0] == edge[1]:
                raise ValueError(f"unexpected self-loop at line {line_number}")
            edges.append(edge)
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate edges found after remapping")
    return edges


def _rows(records: tuple[CascadeRecord, ...]) -> list[dict[str, Any]]:
    return [
        {"cascade": list(record.cascade), "timestamp": list(record.timestamp)}
        for record in records
    ]


def prepare(source: Path, output: Path) -> dict[str, Any]:
    mapping = load_user_mapping(source)
    records, filtering = load_records(source, mapping)
    split = chronological_split(records, ratios=(0.7, 0.1, 0.2))
    edges = remap_edges(source, mapping)

    output.mkdir(parents=True, exist_ok=True)
    for name, partition in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        (output / f"cascade_{name}.json").write_text(
            json.dumps(_rows(partition), indent=2),
            encoding="utf-8",
        )
    (output / "graph.txt").write_text(
        "".join(f"{source_node},{target_node}\n" for source_node, target_node in edges),
        encoding="utf-8",
    )

    num_nodes = len(mapping) - 2
    manifest: dict[str, Any] = {
        "status": "prepared_untouched_temporal_confirmation",
        "dataset": "memetracker",
        "source": {
            "directory": str(source.resolve()),
            "cascades_sha256": file_sha256(source / "cascades.txt"),
            "edges_sha256": file_sha256(source / "edges.txt"),
            "u2idx_sha256": file_sha256(source / "u2idx.pickle"),
        },
        "timestamp_parser": {
            "rule": "insert decimal point after the first six digits",
            "all_retained_cascades_nondecreasing": True,
        },
        "filter": {"minimum_length": 5, "maximum_length": 500, **filtering},
        "split": {
            "ratios": [0.7, 0.1, 0.2],
            "chronological": True,
            "timestamp_ties_preserved": True,
            "strict_train_valid_only_during_selection": True,
        },
        "counts": {
            "all": len(records),
            "train": len(split.train),
            "valid": len(split.valid),
            "test": len(split.test),
            "nodes": num_nodes,
            "edges": len(edges),
        },
    }
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["outputs"] = {
        name: file_sha256(output / name)
        for name in ("cascade_train.json", "cascade_valid.json", "cascade_test.json", "graph.txt")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
