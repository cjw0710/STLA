"""Backbone-agnostic temporal residual for frozen next-user logits."""

from __future__ import annotations

import math

import torch
from torch import nn

from .environment_encoder import EnvironmentEncoder


class TemporalLogitAdapter(nn.Module):
    """Calibrate any frozen anchor using past-only node and prefix features.

    The adapter never consumes an anchor hidden state. Its interface therefore
    works for sequence, graph, and hypergraph predictors that expose logits.
    The final layer is zero-initialized, so construction exactly reproduces the
    anchor ranking.
    """

    base_node_feature_dim = 5
    sample_feature_dim = 8
    ablations = ("full", "no_environment", "no_prefix", "historical_only")

    def __init__(
        self,
        num_nodes: int,
        *,
        context_dim: int = 16,
        hidden_dim: int = 64,
        environment_hidden_dim: int = 32,
        dropout: float = 0.2,
        node_rank: int = 0,
    ) -> None:
        super().__init__()
        if min(num_nodes, context_dim, hidden_dim, environment_hidden_dim) < 1:
            raise ValueError("adapter dimensions must be positive")
        if node_rank < 0:
            raise ValueError("node_rank cannot be negative")
        self.num_nodes = num_nodes
        self.node_rank = node_rank
        self.node_feature_dim = self.base_node_feature_dim + node_rank
        self.node_codes = (
            nn.Embedding(num_nodes, node_rank)
            if node_rank
            else None
        )
        if self.node_codes is not None:
            nn.init.normal_(self.node_codes.weight, std=0.02)
        self.environment_encoder = EnvironmentEncoder(
            context_dim=context_dim,
            hidden_dim=environment_hidden_dim,
            dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(context_dim + self.sample_feature_dim),
            nn.Linear(context_dim + self.sample_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.node_feature_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _standardize(values: torch.Tensor) -> torch.Tensor:
        values = values.float().flatten()
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)

    def temporal_node_features(
        self,
        historical_popularity: torch.Tensor,
        recent_popularity: torch.Tensor,
        *,
        ablation: str = "full",
    ) -> torch.Tensor:
        self._validate_ablation(ablation)
        historical = historical_popularity.float().flatten()
        recent = recent_popularity.float().flatten()
        if historical.numel() != self.num_nodes or recent.shape != historical.shape:
            raise ValueError("popularity vectors must contain one value per node")
        if bool(torch.any(historical < 0)) or bool(torch.any(recent < 0)):
            raise ValueError("popularity cannot be negative")
        historical_z = self._standardize(torch.log1p(historical))
        recent_z = self._standardize(torch.log1p(recent))
        dormant = ((historical > 0) & (recent == 0)).float()
        emerging = (historical == 0).float()
        fixed_features = torch.stack(
            [historical_z, recent_z, recent_z - historical_z, dormant, emerging],
            dim=1,
        )
        if ablation == "historical_only":
            fixed_features = fixed_features.clone()
            fixed_features[:, 1:] = 0.0
        if self.node_codes is None:
            return fixed_features
        node_codes = self.node_codes.weight
        if ablation == "historical_only":
            node_codes = torch.zeros_like(node_codes)
        return torch.cat([fixed_features, node_codes], dim=1)

    @classmethod
    def _validate_ablation(cls, ablation: str) -> None:
        if ablation not in cls.ablations:
            raise ValueError(
                f"unknown adapter ablation {ablation!r}; expected one of {cls.ablations}"
            )

    def prefix_features(
        self,
        shifted_sequence: torch.Tensor,
        timestamps: torch.Tensor,
        historical_popularity: torch.Tensor,
        recent_popularity: torch.Tensor,
        *,
        input_id_offset: int = 2,
    ) -> torch.Tensor:
        """Return one eight-feature descriptor per next-user position."""

        if shifted_sequence.ndim != 2 or timestamps.shape != shifted_sequence.shape:
            raise ValueError("sequence and timestamps must have equal [B, L] shapes")
        if shifted_sequence.shape[1] < 2:
            raise ValueError("a batch must contain at least one prediction position")
        if input_id_offset < 0:
            raise ValueError("input_id_offset cannot be negative")
        inputs = shifted_sequence[:, :-1]
        input_times = timestamps[:, :-1]
        valid = inputs.ne(0)
        node_ids = (inputs - input_id_offset).clamp(min=0, max=self.num_nodes - 1)
        historical_log = torch.log1p(historical_popularity.float().clamp_min(0.0))
        recent_log = torch.log1p(recent_popularity.float().clamp_min(0.0))
        historical_values = historical_log[node_ids].masked_fill(~valid, 0.0)
        recent_values = recent_log[node_ids].masked_fill(~valid, 0.0)
        counts = valid.cumsum(dim=1).clamp_min(1).float()
        historical_mean = historical_values.cumsum(dim=1) / counts
        recent_mean = recent_values.cumsum(dim=1) / counts
        negative = torch.finfo(historical_values.dtype).min
        historical_max = historical_values.masked_fill(~valid, negative).cummax(dim=1).values
        recent_max = recent_values.masked_fill(~valid, negative).cummax(dim=1).values
        historical_max = torch.where(torch.isfinite(historical_max), historical_max, torch.zeros_like(historical_max))
        recent_max = torch.where(torch.isfinite(recent_max), recent_max, torch.zeros_like(recent_max))

        dormant_nodes = ((historical_popularity > 0) & (recent_popularity == 0)).float()
        emerging_nodes = (historical_popularity == 0).float()
        dormant_fraction = (dormant_nodes[node_ids] * valid).cumsum(dim=1) / counts
        emerging_fraction = (emerging_nodes[node_ids] * valid).cumsum(dim=1) / counts

        first_time = input_times[:, :1]
        elapsed = (input_times - first_time).clamp_min(0.0).masked_fill(~valid, 0.0)
        duration = torch.tanh(torch.log1p(elapsed) / 20.0)
        length_scale = math.log1p(max(1, inputs.shape[1]))
        relative_length = torch.log1p(counts) / length_scale
        return torch.stack(
            [
                relative_length,
                duration,
                historical_mean,
                historical_max,
                recent_mean,
                recent_max,
                dormant_fraction,
                emerging_fraction,
            ],
            dim=-1,
        )

    def forward(
        self,
        anchor_logits: torch.Tensor,
        shifted_sequence: torch.Tensor,
        timestamps: torch.Tensor,
        environment_features: torch.Tensor,
        historical_popularity: torch.Tensor,
        recent_popularity: torch.Tensor,
        *,
        input_id_offset: int = 2,
        ablation: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_ablation(ablation)
        if anchor_logits.ndim != 2 or anchor_logits.shape[1] != self.num_nodes:
            raise ValueError("anchor_logits must have shape [predictions, num_nodes]")
        sample_features = self.prefix_features(
            shifted_sequence,
            timestamps,
            historical_popularity,
            recent_popularity,
            input_id_offset=input_id_offset,
        ).reshape(-1, self.sample_feature_dim)
        if ablation == "no_prefix":
            sample_features = torch.zeros_like(sample_features)
        if sample_features.shape[0] != anchor_logits.shape[0]:
            raise ValueError("anchor logits do not match sequence prediction positions")
        context = self.environment_encoder(environment_features).reshape(1, -1)
        if ablation == "no_environment":
            context = torch.zeros_like(context)
        context = context.expand(sample_features.shape[0], -1)
        coefficients = self.gate(torch.cat([sample_features, context], dim=1))
        node_features = self.temporal_node_features(
            historical_popularity,
            recent_popularity,
            ablation=ablation,
        )
        residual = coefficients @ node_features.transpose(0, 1)
        return anchor_logits + residual, residual
