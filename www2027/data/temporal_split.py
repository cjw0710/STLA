"""Leakage-safe chronological splitting of diffusion cascades.

All boundaries are selected between timestamp groups. Cascades with the same
start timestamp therefore never straddle two partitions or environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CascadeRecord:
    """One immutable diffusion cascade with provenance information."""

    cascade: tuple[int, ...]
    timestamp: tuple[float, ...]
    source_split: str
    source_index: int

    def __post_init__(self) -> None:
        if not self.cascade:
            raise ValueError("a cascade must contain at least one node")
        if len(self.cascade) != len(self.timestamp):
            raise ValueError("cascade and timestamp must have equal lengths")
        if any(b < a for a, b in zip(self.timestamp, self.timestamp[1:])):
            raise ValueError("timestamps inside a cascade must be nondecreasing")

    @property
    def start_time(self) -> float:
        return self.timestamp[0]


@dataclass(frozen=True)
class TemporalEnvironment:
    """A contiguous group of cascades on the global time axis."""

    name: str
    records: tuple[CascadeRecord, ...]

    @property
    def start_time(self) -> float:
        return self.records[0].start_time

    @property
    def end_time(self) -> float:
        return self.records[-1].start_time


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train/validation/test partitions."""

    train: tuple[CascadeRecord, ...]
    valid: tuple[CascadeRecord, ...]
    test: tuple[CascadeRecord, ...]


def _sorted_records(records: Iterable[CascadeRecord]) -> list[CascadeRecord]:
    return sorted(
        records,
        key=lambda item: (item.start_time, item.source_split, item.source_index),
    )


def _group_by_start_time(records: Sequence[CascadeRecord]) -> list[list[CascadeRecord]]:
    groups: list[list[CascadeRecord]] = []
    for record in records:
        if not groups or groups[-1][0].start_time != record.start_time:
            groups.append([record])
        else:
            groups[-1].append(record)
    return groups


def _partition_without_tie_breaks(
    records: Iterable[CascadeRecord],
    weights: Sequence[float],
) -> list[tuple[CascadeRecord, ...]]:
    """Divide records approximately by count while preserving timestamp ties."""

    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("partition weights must be nonempty and positive")

    ordered = _sorted_records(records)
    if not ordered:
        raise ValueError("cannot partition an empty collection")

    groups = _group_by_start_time(ordered)
    if len(groups) < len(weights):
        raise ValueError(
            f"need at least {len(weights)} distinct start times; found {len(groups)}"
        )

    total_weight = float(sum(weights))
    cumulative_targets = [
        len(ordered) * sum(weights[:index]) / total_weight
        for index in range(1, len(weights))
    ]

    boundaries: list[int] = []
    cumulative_count = 0
    group_index = 0
    for partition_index, target in enumerate(cumulative_targets):
        groups_left_after_boundary = len(weights) - partition_index - 1
        maximum_group_index = len(groups) - groups_left_after_boundary

        while group_index < maximum_group_index:
            next_count = cumulative_count + len(groups[group_index])
            if group_index > partition_index and abs(cumulative_count - target) <= abs(next_count - target):
                break
            cumulative_count = next_count
            group_index += 1

        # Guarantee a nonempty current partition even when a large tie group
        # makes the approximate target awkward.
        minimum_group_index = partition_index + 1
        while group_index < minimum_group_index:
            cumulative_count += len(groups[group_index])
            group_index += 1
        boundaries.append(group_index)

    partitions: list[tuple[CascadeRecord, ...]] = []
    previous = 0
    for boundary in [*boundaries, len(groups)]:
        partitions.append(tuple(item for group in groups[previous:boundary] for item in group))
        previous = boundary

    return partitions


def load_cascades(
    dataset_dir: str | Path,
    split_names: Sequence[str] = ("train", "valid", "test"),
) -> tuple[CascadeRecord, ...]:
    """Load and merge the legacy JSON files while retaining their provenance."""

    dataset_path = Path(dataset_dir)
    records: list[CascadeRecord] = []
    for split_name in split_names:
        file_path = dataset_path / f"cascade_{split_name}.json"
        with file_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            cascades = payload.get("cascade")
            timestamps = payload.get("timestamp")
            if cascades is None or timestamps is None:
                raise ValueError(f"{file_path} must contain cascade and timestamp arrays")
            if len(cascades) != len(timestamps):
                raise ValueError(f"{file_path}: cascade/timestamp row counts differ")
            rows = [
                {"cascade": cascade, "timestamp": timestamp}
                for cascade, timestamp in zip(cascades, timestamps)
            ]
        else:
            raise ValueError(f"{file_path}: unsupported JSON layout")

        for source_index, row in enumerate(rows):
            if not isinstance(row, dict) or "cascade" not in row or "timestamp" not in row:
                raise ValueError(f"{file_path}: row {source_index} is malformed")
            records.append(
                CascadeRecord(
                    cascade=tuple(int(node) for node in row["cascade"]),
                    timestamp=tuple(float(value) for value in row["timestamp"]),
                    source_split=split_name,
                    source_index=source_index,
                )
            )

    return tuple(_sorted_records(records))


def chronological_split(
    records: Iterable[CascadeRecord],
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
) -> TemporalSplit:
    """Create strict chronological train/validation/test partitions."""

    if len(ratios) != 3:
        raise ValueError("chronological_split requires exactly three ratios")
    train, valid, test = _partition_without_tie_breaks(records, ratios)
    return TemporalSplit(train=train, valid=valid, test=test)


def make_temporal_environments(
    records: Iterable[CascadeRecord],
    n_environments: int,
    prefix: str = "env",
) -> tuple[TemporalEnvironment, ...]:
    """Build contiguous, approximately equal-count temporal environments."""

    if n_environments < 1:
        raise ValueError("n_environments must be positive")
    partitions = _partition_without_tie_breaks(records, [1.0] * n_environments)
    return tuple(
        TemporalEnvironment(name=f"{prefix}_{index}", records=partition)
        for index, partition in enumerate(partitions)
    )
