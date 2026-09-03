"""Environment-conditioned low-rank mask evaluated only on sparse edges."""

from __future__ import annotations

import math

import torch
from torch import nn


class DynamicLowRankMask(nn.Module):
    """Split observed edge weights into stable and shortcut channels.

    The module stores O(NK) node factors and evaluates O(EK) scores for the
    supplied edges. It never constructs an N-by-N mask.
    """

    def __init__(
        self,
        num_nodes: int,
        rank: int,
        context_dim: int,
        *,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        if min(num_nodes, rank, context_dim) < 1:
            raise ValueError("num_nodes, rank, and context_dim must be positive")
        self.num_nodes = num_nodes
        self.rank = rank
        self.context_dim = context_dim
        self.left = nn.Embedding(num_nodes, rank)
        self.right = nn.Embedding(num_nodes, rank)
        self.left_scale = nn.Linear(context_dim, rank)
        self.right_scale = nn.Linear(context_dim, rank)
        self.environment_bias = nn.Linear(context_dim, 1)
        if init_std is None:
            init_std = 1.0 / math.sqrt(rank)
        nn.init.normal_(self.left.weight, std=init_std)
        nn.init.normal_(self.right.weight, std=init_std)
        nn.init.xavier_uniform_(self.left_scale.weight, gain=0.1)
        nn.init.zeros_(self.left_scale.bias)
        nn.init.xavier_uniform_(self.right_scale.weight, gain=0.1)
        nn.init.zeros_(self.right_scale.bias)
        nn.init.xavier_uniform_(self.environment_bias.weight, gain=0.05)
        nn.init.zeros_(self.environment_bias.bias)

    def _check_edge_index(self, edge_index: torch.Tensor) -> None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must use an integer dtype")
        if edge_index.numel() and (
            bool(torch.any(edge_index < 0))
            or bool(torch.any(edge_index >= self.num_nodes))
        ):
            raise ValueError("edge_index contains an out-of-range node id")

    def forward(self, edge_index: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return a stable-channel mask value in [0, 1] for each edge."""

        self._check_edge_index(edge_index)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(f"context must have shape [1|E, {self.context_dim}]")

        num_edges = edge_index.shape[1]
        if context.shape[0] not in (1, num_edges):
            raise ValueError("context batch must contain one row or one row per edge")

        source, destination = edge_index
        left = self.left(source)
        right = self.right(destination)
        left_scale = 1.0 + torch.tanh(self.left_scale(context))
        right_scale = 1.0 + torch.tanh(self.right_scale(context))
        score = ((left * left_scale) * (right * right_scale)).sum(dim=-1)
        score = score / math.sqrt(self.rank)
        score = score + self.environment_bias(context).squeeze(-1)
        return torch.sigmoid(score)

    def split_weights(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stable weights, shortcut weights, and the learned mask."""

        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have one value per edge")
        mask = self(edge_index, context)
        stable = edge_weight * mask
        shortcut = edge_weight * (1.0 - mask)
        return stable, shortcut, mask
