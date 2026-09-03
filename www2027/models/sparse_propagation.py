"""Sparse weighted message passing used by DriftDiff."""

from __future__ import annotations

import torch
from torch import nn


class SparseGraphPropagation(nn.Module):
    """One residual message-passing layer with destination normalization."""

    def __init__(self, dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.message = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dimension),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [N, D]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have one value per edge")

        source, destination = edge_index
        messages = self.message(node_features[source]) * edge_weight.unsqueeze(-1)
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, destination, messages)
        normalizer = node_features.new_zeros(node_features.shape[0])
        normalizer.index_add_(0, destination, edge_weight.abs())
        aggregated = aggregated / normalizer.clamp_min(1e-12).unsqueeze(-1)
        return self.output(torch.cat([node_features, aggregated], dim=-1))
