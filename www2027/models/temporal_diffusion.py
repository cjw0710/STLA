"""End-to-end diffusion predictor conditioned on historical environments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .dynamic_low_rank_mask import DynamicLowRankMask
from .environment_encoder import EnvironmentEncoder
from .sparse_propagation import SparseGraphPropagation


@dataclass
class DiffusionOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    shortcut_logits: torch.Tensor
    temporal_prior_logits: torch.Tensor
    stable_representation: torch.Tensor
    shortcut_representation: torch.Tensor
    edge_mask: torch.Tensor


class TemporalDiffusionModel(nn.Module):
    """Predict the next activated user from a cascade prefix.

    The graph mask changes with the historical environment context, while all
    edge scores are evaluated sparsely. User id ``num_nodes`` is reserved for
    padding so real user zero remains representable.
    """

    def __init__(
        self,
        num_nodes: int,
        dimension: int = 64,
        rank: int = 16,
        context_dim: int = 16,
        environment_hidden_dim: int = 32,
        dropout: float = 0.2,
        mask_mode: str = "dynamic",
        prior_mode: str = "none",
    ) -> None:
        super().__init__()
        if min(num_nodes, dimension, rank, context_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        self.num_nodes = num_nodes
        self.padding_id = num_nodes
        self.dimension = dimension
        self.logit_scale = 1.0 / math.sqrt(dimension)
        if mask_mode not in {"dynamic", "static", "none"}:
            raise ValueError("mask_mode must be dynamic, static, or none")
        if prior_mode not in {"none", "temporal"}:
            raise ValueError("prior_mode must be none or temporal")
        self.mask_mode = mask_mode
        self.prior_mode = prior_mode
        self.user_embedding = nn.Embedding(num_nodes + 1, dimension, padding_idx=self.padding_id)
        self.environment_encoder = EnvironmentEncoder(
            context_dim=context_dim,
            hidden_dim=environment_hidden_dim,
            dropout=dropout,
        )
        self.graph_mask = DynamicLowRankMask(num_nodes, rank, context_dim)
        self.stable_propagation = SparseGraphPropagation(dimension, dropout)
        self.shortcut_propagation = SparseGraphPropagation(dimension, dropout)
        self.time_projection = nn.Sequential(
            nn.Linear(2, dimension),
            nn.Tanh(),
        )
        self.sequence_encoder = nn.GRU(
            input_size=dimension,
            hidden_size=dimension,
            batch_first=True,
        )
        self.sequence_norm = nn.LayerNorm(dimension)
        self.shortcut_projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.temporal_prior_gate = nn.Sequential(
            nn.Linear(dimension + context_dim, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 5),
        )
        # Start exactly at the no-prior predictor so a paired ablation measures
        # learned temporal residuals rather than an arbitrary score-scale shift.
        nn.init.zeros_(self.temporal_prior_gate[-1].weight)
        nn.init.zeros_(self.temporal_prior_gate[-1].bias)
        self.output_bias = nn.Parameter(torch.zeros(num_nodes))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _standardize(values: torch.Tensor) -> torch.Tensor:
        values = values.float().flatten()
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)

    def _temporal_node_features(
        self,
        historical_popularity: torch.Tensor,
        recent_popularity: torch.Tensor,
    ) -> torch.Tensor:
        if historical_popularity.numel() != self.num_nodes:
            raise ValueError("historical popularity must contain one value per node")
        if recent_popularity.shape != historical_popularity.shape:
            raise ValueError("recent and historical popularity shapes must match")
        historical_log = torch.log1p(historical_popularity.clamp_min(0.0))
        recent_log = torch.log1p(recent_popularity.clamp_min(0.0))
        historical_z = self._standardize(historical_log)
        recent_z = self._standardize(recent_log)
        dormant = ((historical_popularity > 0) & (recent_popularity == 0)).float()
        emerging = (historical_popularity == 0).float()
        return torch.stack(
            [
                historical_z,
                recent_z,
                recent_z - historical_z,
                dormant,
                emerging,
            ],
            dim=1,
        )

    def _time_features(self, elapsed: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        log_elapsed = torch.log1p(elapsed.clamp_min(0.0))
        final_indices = (lengths - 1).clamp_min(0).unsqueeze(1)
        log_duration = log_elapsed.gather(1, final_indices).clamp_min(1e-6)
        relative = log_elapsed / log_duration
        absolute = torch.tanh(log_duration / 20.0).expand_as(relative)
        return torch.stack([relative, absolute], dim=-1)

    def _mask_seen_users(self, logits: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        seen = torch.zeros(
            prefix.shape[0],
            self.num_nodes + 1,
            dtype=torch.bool,
            device=prefix.device,
        )
        seen.scatter_(1, prefix, True)
        return logits.masked_fill(seen[:, : self.num_nodes], float("-inf"))

    def forward(
        self,
        prefix: torch.Tensor,
        elapsed: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        environment_features: torch.Tensor,
        historical_popularity: torch.Tensor | None = None,
        recent_popularity: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        if prefix.ndim != 2 or elapsed.shape != prefix.shape:
            raise ValueError("prefix and elapsed must have equal [B, L] shapes")
        if lengths.ndim != 1 or lengths.shape[0] != prefix.shape[0]:
            raise ValueError("lengths must have shape [B]")

        context = self.environment_encoder(environment_features)
        if self.mask_mode == "static":
            # A zero context retains one globally learned low-rank mask while
            # removing all temporal conditioning.
            mask_context = torch.zeros_like(context)
            stable_weights, shortcut_weights, edge_mask = self.graph_mask.split_weights(
                edge_index,
                edge_weight,
                mask_context,
            )
        elif self.mask_mode == "none":
            edge_mask = torch.ones_like(edge_weight)
            stable_weights = edge_weight
            shortcut_weights = torch.zeros_like(edge_weight)
        else:
            stable_weights, shortcut_weights, edge_mask = self.graph_mask.split_weights(
                edge_index,
                edge_weight,
                context,
            )
        base_nodes = self.user_embedding.weight[: self.num_nodes]
        stable_nodes = base_nodes + self.stable_propagation(
            base_nodes, edge_index, stable_weights
        )
        shortcut_nodes = base_nodes + self.shortcut_propagation(
            base_nodes, edge_index, shortcut_weights
        )

        zero_padding = base_nodes.new_zeros(1, self.dimension)
        stable_lookup = torch.cat([stable_nodes, zero_padding], dim=0)
        shortcut_lookup = torch.cat([shortcut_nodes, zero_padding], dim=0)
        sequence = stable_lookup[prefix] + self.time_projection(
            self._time_features(elapsed, lengths)
        )
        sequence = self.dropout(sequence)
        packed = pack_padded_sequence(
            sequence,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.sequence_encoder(packed)
        stable_representation = self.sequence_norm(hidden[-1])

        valid = (prefix != self.padding_id).unsqueeze(-1)
        shortcut_sequence = shortcut_lookup[prefix] * valid
        shortcut_representation = shortcut_sequence.sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        shortcut_representation = self.shortcut_projection(shortcut_representation)

        base_logits = (
            stable_representation @ stable_nodes.transpose(0, 1)
        ) * self.logit_scale + self.output_bias
        temporal_prior_logits = torch.zeros_like(base_logits)
        if self.prior_mode == "temporal":
            if historical_popularity is None or recent_popularity is None:
                raise ValueError(
                    "temporal prior requires historical and recent popularity"
                )
            node_features = self._temporal_node_features(
                historical_popularity,
                recent_popularity,
            )
            repeated_context = context.reshape(1, -1).expand(prefix.shape[0], -1)
            coefficients = self.temporal_prior_gate(
                torch.cat([stable_representation, repeated_context], dim=1)
            )
            temporal_prior_logits = coefficients @ node_features.transpose(0, 1)
        logits = base_logits + temporal_prior_logits
        base_logits = self._mask_seen_users(base_logits, prefix)
        logits = self._mask_seen_users(logits, prefix)
        shortcut_logits = (
            shortcut_representation @ shortcut_nodes.transpose(0, 1)
        ) * self.logit_scale
        return DiffusionOutput(
            logits=logits,
            base_logits=base_logits,
            shortcut_logits=shortcut_logits,
            temporal_prior_logits=temporal_prior_logits,
            stable_representation=stable_representation,
            shortcut_representation=shortcut_representation,
            edge_mask=edge_mask,
        )
