"""Prefix-to-next-user examples without conflating user zero with padding."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .temporal_split import CascadeRecord


class NextUserDataset(Dataset[dict[str, torch.Tensor]]):
    """Expand cascades into prefix/next-user prediction examples.

    Only the first ``max_prefix_length + 1`` activations of a cascade are used,
    matching the common early diffusion prediction setting. Repeated targets
    already present in their prefix are skipped because next-user ranking masks
    previously activated users.
    """

    def __init__(
        self,
        records: Sequence[CascadeRecord],
        num_nodes: int,
        max_prefix_length: int,
    ) -> None:
        if num_nodes < 1 or max_prefix_length < 1:
            raise ValueError("num_nodes and max_prefix_length must be positive")
        self.records = tuple(records)
        self.num_nodes = num_nodes
        self.padding_id = num_nodes
        self.max_prefix_length = max_prefix_length
        self.examples: list[tuple[int, int]] = []

        for record_index, record in enumerate(self.records):
            upper_bound = min(len(record.cascade), max_prefix_length + 1)
            seen: set[int] = set()
            for target_position in range(1, upper_bound):
                seen.add(record.cascade[target_position - 1])
                target = record.cascade[target_position]
                if target not in seen:
                    self.examples.append((record_index, target_position))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, target_position = self.examples[index]
        record = self.records[record_index]
        prefix_nodes = record.cascade[:target_position]
        prefix_times = record.timestamp[:target_position]
        length = len(prefix_nodes)

        prefix = torch.full(
            (self.max_prefix_length,),
            self.padding_id,
            dtype=torch.long,
        )
        elapsed = torch.zeros(self.max_prefix_length, dtype=torch.float32)
        prefix[:length] = torch.tensor(prefix_nodes, dtype=torch.long)
        start_time = prefix_times[0]
        elapsed[:length] = torch.tensor(
            [max(0.0, timestamp - start_time) for timestamp in prefix_times],
            dtype=torch.float32,
        )
        return {
            "prefix": prefix,
            "elapsed": elapsed,
            "length": torch.tensor(length, dtype=torch.long),
            "target": torch.tensor(record.cascade[target_position], dtype=torch.long),
        }
