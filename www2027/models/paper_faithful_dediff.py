"""Executable sequence and objective stages for the DeDiff PDF equations.

The implementation consumes the sparse graph output from
``PaperGraphDisentangler`` and follows the printed STAN, next-user dot-product,
and bias-distribution objectives.  Choices needed to make inconsistent or
underspecified PDF notation executable are frozen in
``config/paper_faithful_v1.json`` and surfaced as constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .paper_graph_disentangler import PaperGraphDisentangler, PaperGraphOutput


@dataclass(frozen=True)
class PaperDeDiffOutput:
    """Predictions and auxiliary graph terms required by Eq. (17)."""

    logits: torch.Tensor
    bias_logits: torch.Tensor
    causal_hidden: torch.Tensor
    bias_hidden: torch.Tensor
    causal_nodes: torch.Tensor
    bias_nodes: torch.Tensor
    inter_view_loss: torch.Tensor
    disagreement_loss: torch.Tensor
    graph: PaperGraphOutput


@dataclass(frozen=True)
class PaperDeDiffLoss:
    total: torch.Tensor
    prediction: torch.Tensor
    bias: torch.Tensor
    disagreement: torch.Tensor
    inter_view: torch.Tensor


class PaperSTAN(nn.Module):
    """Social-temporal multi-head attention from Eqs. (11)-(14).

    The PDF declares one ``d x d`` query/key/value projection per head but also
    writes ``d' = d/B``.  For the reported ``d=64, B=10``, the latter is not an
    integer.  This module follows the explicit projection shapes: every head
    has a configurable full head dimension and attention is scaled by its
    square root.
    """

    def __init__(
        self,
        dimension: int,
        heads: int = 10,
        *,
        head_dimension: int | None = None,
        social_distance_bins: int = 32,
        time_interval_bins: int = 32,
        prior_dimension: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if min(
            dimension,
            heads,
            social_distance_bins,
            time_interval_bins,
            prior_dimension,
        ) < 1:
            raise ValueError("all STAN dimensions must be positive")
        if head_dimension is None:
            head_dimension = dimension
        if head_dimension < 1:
            raise ValueError("head_dimension must be positive")
        self.dimension = dimension
        self.heads = heads
        self.head_dimension = head_dimension
        self.social_distance_bins = social_distance_bins
        self.time_interval_bins = time_interval_bins
        projected_dimension = heads * head_dimension
        self.query = nn.Linear(dimension, projected_dimension, bias=False)
        self.key = nn.Linear(dimension, projected_dimension, bias=False)
        self.value = nn.Linear(dimension, projected_dimension, bias=False)
        self.output = nn.Linear(projected_dimension, dimension, bias=False)
        self.social_distance_embedding = nn.Embedding(
            social_distance_bins,
            prior_dimension,
        )
        self.time_interval_embedding = nn.Embedding(
            time_interval_bins,
            prior_dimension,
        )
        self.social_temporal_bias = nn.Sequential(
            nn.Linear(2 * prior_dimension, prior_dimension),
            nn.ReLU(),
            nn.Linear(prior_dimension, heads),
        )
        self.feed_forward_1 = nn.Linear(dimension, dimension)
        self.feed_forward_2 = nn.Linear(dimension, dimension)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _pairwise_bias(
        self,
        elapsed: torch.Tensor,
        social_distance: torch.Tensor,
    ) -> torch.Tensor:
        if elapsed.ndim != 2:
            raise ValueError("elapsed must have shape [batch, length]")
        batch, length = elapsed.shape
        if social_distance.shape != (batch, length, length):
            raise ValueError("social_distance must have shape [batch, length, length]")
        distance = torch.nan_to_num(
            social_distance.float(),
            nan=float(self.social_distance_bins - 1),
            posinf=float(self.social_distance_bins - 1),
            neginf=0.0,
        ).round().long()
        distance = distance.clamp(0, self.social_distance_bins - 1)
        interval = (elapsed.unsqueeze(2) - elapsed.unsqueeze(1)).abs()
        # Log buckets keep long-tailed time gaps representable without using
        # any information beyond the supplied cascade prefix.
        interval_bucket = torch.floor(torch.log1p(interval)).long()
        interval_bucket = interval_bucket.clamp(0, self.time_interval_bins - 1)
        social = self.social_distance_embedding(distance)
        temporal = self.time_interval_embedding(interval_bucket)
        bias = self.social_temporal_bias(torch.cat([social, temporal], dim=-1))
        return bias.permute(0, 3, 1, 2)

    def forward(
        self,
        sequence: torch.Tensor,
        elapsed: torch.Tensor,
        social_distance: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, length, dimension]")
        batch, length, dimension = sequence.shape
        if dimension != self.dimension or elapsed.shape != (batch, length):
            raise ValueError("sequence and elapsed shapes are incompatible")
        if lengths.ndim != 1 or lengths.shape[0] != batch:
            raise ValueError("lengths must have shape [batch]")
        if bool(torch.any(lengths < 1)) or bool(torch.any(lengths > length)):
            raise ValueError("lengths must fall inside the sequence extent")

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, length, self.heads, self.head_dimension).transpose(1, 2)

        query = heads(self.query(sequence))
        key = heads(self.key(sequence))
        value = heads(self.value(sequence))
        score = query @ key.transpose(-1, -2)
        score = score / math.sqrt(float(self.head_dimension))
        score = score + self._pairwise_bias(elapsed, social_distance)

        positions = torch.arange(length, device=sequence.device)
        future = positions.unsqueeze(0) > positions.unsqueeze(1)
        valid_key = positions.reshape(1, 1, 1, length) < lengths.reshape(batch, 1, 1, 1)
        score = score.masked_fill(future.reshape(1, 1, length, length), float("-inf"))
        score = score.masked_fill(~valid_key, float("-inf"))
        attention = torch.softmax(score, dim=-1)
        attention = self.attention_dropout(attention)
        attended = attention @ value
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        attended = self.output_dropout(self.output(attended))
        hidden = self.feed_forward_2(F.relu(self.feed_forward_1(attended)))
        valid_query = positions.reshape(1, length, 1) < lengths.reshape(batch, 1, 1)
        return hidden.masked_fill(~valid_query, 0.0)


class PaperFaithfulDeDiff(nn.Module):
    """Sparse graph-faithful DeDiff plus paper-formula sequence branches."""

    def __init__(
        self,
        num_nodes: int,
        dimension: int = 64,
        rank: int = 16,
        gcn_layers: int = 1,
        attention_heads: int = 10,
        attention_head_dimension: int | None = None,
        social_distance_bins: int = 32,
        time_interval_bins: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        self.num_nodes = num_nodes
        self.padding_id = num_nodes
        self.graph_disentangler = PaperGraphDisentangler(
            num_nodes=num_nodes,
            dimension=dimension,
            rank=rank,
            gcn_layers=gcn_layers,
        )
        self.causal_view_fusion = nn.Linear(2 * dimension, dimension)
        self.bias_view_fusion = nn.Linear(2 * dimension, dimension)
        stan_arguments = dict(
            dimension=dimension,
            heads=attention_heads,
            head_dimension=attention_head_dimension,
            social_distance_bins=social_distance_bins,
            time_interval_bins=time_interval_bins,
            dropout=dropout,
        )
        self.causal_stan = PaperSTAN(**stan_arguments)
        self.bias_stan = PaperSTAN(**stan_arguments)

    def _mask_seen_users(self, logits: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        seen = torch.zeros(
            prefix.shape[0],
            self.num_nodes + 1,
            device=prefix.device,
            dtype=torch.bool,
        )
        seen.scatter_(1, prefix, True)
        return logits.masked_fill(seen[:, : self.num_nodes], float("-inf"))

    def forward(
        self,
        prefix: torch.Tensor,
        elapsed: torch.Tensor,
        lengths: torch.Tensor,
        social_distance: torch.Tensor,
        interaction_edge_index: torch.Tensor,
        interaction_edge_weight: torch.Tensor,
        social_edge_index: torch.Tensor,
        social_edge_weight: torch.Tensor,
        *,
        hinged_disagreement: bool = False,
    ) -> PaperDeDiffOutput:
        if prefix.ndim != 2 or elapsed.shape != prefix.shape:
            raise ValueError("prefix and elapsed must have equal [batch, length] shapes")
        if prefix.numel() and (
            bool(torch.any(prefix < 0)) or bool(torch.any(prefix > self.padding_id))
        ):
            raise ValueError("prefix contains an invalid user or padding id")
        graph = self.graph_disentangler(
            interaction_edge_index,
            interaction_edge_weight,
            social_edge_index,
            social_edge_weight,
        )
        inter_view_loss, disagreement_loss = self.graph_disentangler.disentanglement_losses(
            graph,
            hinged_disagreement=hinged_disagreement,
        )
        causal_nodes = self.causal_view_fusion(
            torch.cat([graph.causal_interaction, graph.causal_social], dim=-1)
        )
        bias_nodes = self.bias_view_fusion(
            torch.cat([graph.bias_interaction, graph.bias_social], dim=-1)
        )
        zero = causal_nodes.new_zeros(1, causal_nodes.shape[1])
        causal_lookup = torch.cat([causal_nodes, zero], dim=0)
        bias_lookup = torch.cat([bias_nodes, zero], dim=0)
        causal_sequence = causal_lookup[prefix]
        bias_sequence = bias_lookup[prefix]
        causal_context = self.causal_stan(
            causal_sequence,
            elapsed,
            social_distance,
            lengths,
        )
        bias_context = self.bias_stan(
            bias_sequence,
            elapsed,
            social_distance,
            lengths,
        )
        final_index = (lengths - 1).reshape(-1, 1, 1).expand(-1, 1, causal_nodes.shape[1])
        causal_hidden = causal_context.gather(1, final_index).squeeze(1)
        bias_hidden = bias_context.gather(1, final_index).squeeze(1)
        logits = causal_hidden @ causal_nodes.transpose(0, 1)
        logits = self._mask_seen_users(logits, prefix)
        bias_logits = bias_hidden @ bias_nodes.transpose(0, 1)
        return PaperDeDiffOutput(
            logits=logits,
            bias_logits=bias_logits,
            causal_hidden=causal_hidden,
            bias_hidden=bias_hidden,
            causal_nodes=causal_nodes,
            bias_nodes=bias_nodes,
            inter_view_loss=inter_view_loss,
            disagreement_loss=disagreement_loss,
            graph=graph,
        )


def paper_dediff_loss(
    output: PaperDeDiffOutput,
    target: torch.Tensor,
    popularity_target: torch.Tensor,
    *,
    alpha: float = 1.0,
    lambda_disagreement: float = 1.0,
    lambda_inter_view: float = 1.0,
    kl_direction: str = "prediction_to_target",
    smoothing: float = 1e-8,
) -> PaperDeDiffLoss:
    """Composite objective from Eqs. (15)-(17) with explicit KL direction."""

    if min(alpha, lambda_disagreement, lambda_inter_view) < 0 or smoothing <= 0:
        raise ValueError("loss weights cannot be negative and smoothing must be positive")
    if target.ndim != 1 or target.shape[0] != output.logits.shape[0]:
        raise ValueError("target must have shape [batch]")
    if target.numel() and (
        bool(torch.any(target < 0)) or bool(torch.any(target >= output.logits.shape[1]))
    ):
        raise ValueError("target must contain real-user ids only")
    if popularity_target.ndim != 1 or popularity_target.numel() != output.logits.shape[1]:
        raise ValueError("popularity_target must contain one value per real user")
    if bool(torch.any(popularity_target < 0)):
        raise ValueError("popularity_target cannot be negative")

    prediction = F.cross_entropy(output.logits, target)
    target_distribution = popularity_target.to(output.bias_logits.dtype) + smoothing
    target_distribution = target_distribution / target_distribution.sum().clamp_min(smoothing)
    log_prediction = F.log_softmax(output.bias_logits, dim=-1)
    if kl_direction == "prediction_to_target":
        prediction_distribution = log_prediction.exp()
        log_target = target_distribution.log().reshape(1, -1)
        bias = (
            prediction_distribution * (log_prediction - log_target)
        ).sum(dim=-1).mean()
    elif kl_direction == "target_to_prediction":
        bias = F.kl_div(
            log_prediction,
            target_distribution.reshape(1, -1).expand_as(log_prediction),
            reduction="batchmean",
        )
    else:
        raise ValueError("kl_direction must be prediction_to_target or target_to_prediction")
    total = (
        prediction
        + alpha * bias
        + lambda_disagreement * output.disagreement_loss
        + lambda_inter_view * output.inter_view_loss
    )
    return PaperDeDiffLoss(
        total=total,
        prediction=prediction,
        bias=bias,
        disagreement=output.disagreement_loss,
        inter_view=output.inter_view_loss,
    )
