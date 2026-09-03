"""Compact past-only environment representation."""

from __future__ import annotations

import torch
from torch import nn


def _gini(values: torch.Tensor, eps: float) -> torch.Tensor:
    sorted_values = torch.sort(values.flatten()).values
    total = sorted_values.sum()
    if bool(total <= eps):
        return total.new_zeros(())
    indices = torch.arange(
        1,
        sorted_values.numel() + 1,
        device=values.device,
        dtype=values.dtype,
    )
    n = sorted_values.new_tensor(float(sorted_values.numel()))
    return (2.0 * torch.sum(indices * sorted_values) / (n * total)) - (n + 1.0) / n


def _summarize_environment(
    popularity: torch.Tensor,
    degree: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    popularity = popularity.flatten().float()
    degree = degree.flatten().float()
    if popularity.shape != degree.shape or popularity.numel() == 0:
        raise ValueError("popularity and degree must be nonempty vectors of equal size")
    if bool(torch.any(popularity < 0)) or bool(torch.any(degree < 0)):
        raise ValueError("popularity and degree cannot be negative")

    active = popularity > 0
    active_values = popularity[active]
    if active_values.numel() == 0:
        active_values = popularity.new_zeros(1)

    log_popularity = torch.log1p(active_values)
    total = popularity.sum().clamp_min(eps)
    distribution = popularity / total
    nonzero_distribution = distribution[distribution > 0]
    entropy = -torch.sum(nonzero_distribution * torch.log(nonzero_distribution + eps))
    entropy_normalizer = torch.log(popularity.new_tensor(float(max(popularity.numel(), 2))))

    sorted_popularity = torch.sort(popularity, descending=True).values
    top_one_count = max(1, int(round(0.01 * popularity.numel())))
    top_five_count = max(1, int(round(0.05 * popularity.numel())))

    features = torch.stack(
        [
            active.float().mean(),
            torch.log1p(active.float().sum()),
            log_popularity.mean(),
            log_popularity.std(unbiased=False),
            log_popularity.max(),
            torch.quantile(log_popularity, 0.5),
            torch.quantile(log_popularity, 0.9),
            _gini(popularity, eps),
            entropy / entropy_normalizer,
            sorted_popularity[:top_one_count].sum() / total,
            sorted_popularity[:top_five_count].sum() / total,
            torch.log1p(degree).mean(),
            torch.log1p(degree).std(unbiased=False),
            torch.log1p(degree).max(),
        ]
    )
    return torch.nan_to_num(features)


def build_environment_features(
    popularity: torch.Tensor,
    degree: torch.Tensor,
    recent_popularity: torch.Tensor | None = None,
    recent_degree: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Summarize cumulative and most-recent past environments in 28 dimensions.

    The recent window prevents cumulative statistics from becoming almost
    identical late in training. Every input must still precede the target
    environment. When recent tensors are omitted, cumulative tensors are reused.
    """

    if (recent_popularity is None) != (recent_degree is None):
        raise ValueError("recent_popularity and recent_degree must be supplied together")
    if recent_popularity is None:
        recent_popularity = popularity
        recent_degree = degree
    cumulative = _summarize_environment(popularity, degree, eps)
    recent = _summarize_environment(recent_popularity, recent_degree, eps)
    return torch.cat([cumulative, recent], dim=0)


class EnvironmentEncoder(nn.Module):
    """Map historical environment statistics to a conditioning vector."""

    feature_dim = 28

    def __init__(self, context_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        if context_dim < 1 or hidden_dim < 1:
            raise ValueError("context_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dimension {self.feature_dim}")
        return self.network(features)
