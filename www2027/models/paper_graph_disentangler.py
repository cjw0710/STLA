"""Sparse implementation of the graph stages described in the DeDiff paper.

This module is deliberately separate from the released top-level ``model.py``.
The released implementation learns a dense ``N x N`` operator and applies
``A @ D``.  The paper instead defines an edge-wise low-rank mask
``sigmoid(P Q^T)`` and says that it is evaluated only on observed edges.  The
classes below implement that stated graph decomposition and the four GCN
encoders without ever materializing an ``N x N`` mask.

The sequence encoder and branch-specific prediction losses are intentionally
outside this module.  Keeping the boundary explicit prevents this graph-faithful
component from being mistaken for a complete reproduction of every ambiguous
detail in the PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseGraphSplit:
    """Complementary causal and bias weights on one observed edge set."""

    edge_index: torch.Tensor
    causal_weight: torch.Tensor
    bias_weight: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class PaperGraphOutput:
    """Four graph-view embeddings and their sparse decompositions."""

    causal_interaction: torch.Tensor
    bias_interaction: torch.Tensor
    causal_social: torch.Tensor
    bias_social: torch.Tensor
    interaction_split: SparseGraphSplit
    social_split: SparseGraphSplit


class PaperLowRankEdgeMask(nn.Module):
    """Evaluate ``sigmoid(P Q^T)`` only for supplied observed edges.

    Exactly two ``num_nodes x rank`` factor matrices are learned, matching the
    ``O(NK)`` parameterization in Eq. (2) of the PDF.  One instance can be
    shared by the interaction and social graphs, as stated in the paper.
    """

    def __init__(
        self,
        num_nodes: int,
        rank: int,
        *,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        if num_nodes < 1 or rank < 1:
            raise ValueError("num_nodes and rank must be positive")
        self.num_nodes = num_nodes
        self.rank = rank
        self.left = nn.Parameter(torch.empty(num_nodes, rank))
        self.right = nn.Parameter(torch.empty(num_nodes, rank))
        if init_std is None:
            init_std = 1.0 / math.sqrt(float(rank))
        nn.init.normal_(self.left, std=init_std)
        nn.init.normal_(self.right, std=init_std)

    def _validate_edges(self, edge_index: torch.Tensor) -> None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must use an integer dtype")
        if edge_index.numel() and (
            bool(torch.any(edge_index < 0))
            or bool(torch.any(edge_index >= self.num_nodes))
        ):
            raise ValueError("edge_index contains an out-of-range node id")

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        self._validate_edges(edge_index)
        row, column = edge_index
        # No 1/sqrt(K) score scaling is applied: the PDF defines sigma(PQ^T)
        # directly.  Initialization controls the starting score magnitude.
        score = (self.left[row] * self.right[column]).sum(dim=-1)
        return torch.sigmoid(score)

    def split(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> SparseGraphSplit:
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have one value per observed edge")
        mask = self(edge_index)
        causal = edge_weight * mask
        bias = edge_weight * (1.0 - mask)
        return SparseGraphSplit(edge_index, causal, bias, mask)


class PaperGCNEncoder(nn.Module):
    """Sparse GCN with sigmoid layers and sum fusion from Eqs. (3)-(4)."""

    def __init__(self, dimension: int, layers: int = 1) -> None:
        super().__init__()
        if dimension < 1 or layers < 1:
            raise ValueError("dimension and layers must be positive")
        self.dimension = dimension
        self.layers = nn.ModuleList(
            nn.Linear(dimension, dimension, bias=False) for _ in range(layers)
        )

    @staticmethod
    def _normalized_propagation(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [num_nodes, dimension]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have one value per observed edge")
        if edge_weight.device != features.device or edge_index.device != features.device:
            raise ValueError("features, edge_index, and edge_weight must share a device")

        num_nodes = features.shape[0]
        loops = torch.arange(num_nodes, device=features.device, dtype=torch.long)
        row = torch.cat([edge_index[0].long(), loops])
        column = torch.cat([edge_index[1].long(), loops])
        weight = torch.cat([edge_weight.to(features.dtype), edge_weight.new_ones(num_nodes)])

        # Coalesce repeated observed edges and any pre-existing self-loops so
        # A + I is represented exactly once before degree normalization.
        adjacency = torch.sparse_coo_tensor(
            torch.stack([row, column]),
            weight,
            size=(num_nodes, num_nodes),
            device=features.device,
        ).coalesce()
        row, column = adjacency.indices()
        weight = adjacency.values()
        degree = features.new_zeros(num_nodes)
        degree.index_add_(0, row, weight)
        inverse_sqrt = degree.clamp_min(1e-12).pow(-0.5)
        normalized = weight * inverse_sqrt[row] * inverse_sqrt[column]

        messages = features[column] * normalized.unsqueeze(-1)
        output = torch.zeros_like(features)
        output.index_add_(0, row, messages)
        return output

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        hidden = features
        representations = [hidden]
        for layer in self.layers:
            propagated = self._normalized_propagation(hidden, edge_index, edge_weight)
            hidden = torch.sigmoid(layer(propagated))
            representations.append(hidden)
        return torch.stack(representations, dim=0).sum(dim=0)


class PaperGraphDisentangler(nn.Module):
    """Paper-stated shared mask plus independent encoders for four views."""

    def __init__(
        self,
        num_nodes: int,
        dimension: int = 64,
        rank: int = 16,
        gcn_layers: int = 1,
    ) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.node_embedding = nn.Embedding(num_nodes, dimension)
        self.edge_mask = PaperLowRankEdgeMask(num_nodes, rank)
        self.interaction_causal_encoder = PaperGCNEncoder(dimension, gcn_layers)
        self.interaction_bias_encoder = PaperGCNEncoder(dimension, gcn_layers)
        self.social_causal_encoder = PaperGCNEncoder(dimension, gcn_layers)
        self.social_bias_encoder = PaperGCNEncoder(dimension, gcn_layers)
        self.interaction_projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )
        self.social_projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(
        self,
        interaction_edge_index: torch.Tensor,
        interaction_edge_weight: torch.Tensor,
        social_edge_index: torch.Tensor,
        social_edge_weight: torch.Tensor,
    ) -> PaperGraphOutput:
        interaction = self.edge_mask.split(interaction_edge_index, interaction_edge_weight)
        social = self.edge_mask.split(social_edge_index, social_edge_weight)
        features = self.node_embedding.weight
        return PaperGraphOutput(
            causal_interaction=self.interaction_causal_encoder(
                features, interaction.edge_index, interaction.causal_weight
            ),
            bias_interaction=self.interaction_bias_encoder(
                features, interaction.edge_index, interaction.bias_weight
            ),
            causal_social=self.social_causal_encoder(
                features, social.edge_index, social.causal_weight
            ),
            bias_social=self.social_bias_encoder(
                features, social.edge_index, social.bias_weight
            ),
            interaction_split=interaction,
            social_split=social,
        )

    def disentanglement_losses(
        self,
        output: PaperGraphOutput,
        *,
        margin: float = 1e-5,
        hinged_disagreement: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the paper's inter-view BPR and causal-bias losses.

        ``hinged_disagreement=False`` reproduces Eq. (10) as printed.  The
        optional hinge is exposed for a corrected ablation because the printed
        objective is otherwise not bounded below with respect to embedding
        scale.
        """

        interaction = output.causal_interaction
        social = output.causal_social
        interaction_proxy = interaction.mean(dim=0)

        # Eq. (5) divides the sum of one-hop neighbor embeddings by N rather
        # than normalizing each node by its degree.  Reproduce that definition
        # exactly here; a degree-normalized alternative should be a named
        # ablation, not silently substituted.
        row, column = output.social_split.edge_index
        neighbor_sum = torch.zeros_like(social)
        neighbor_sum.index_add_(0, row.long(), social[column.long()])
        social_proxy = neighbor_sum.sum(dim=0) / float(self.num_nodes)

        projected_interaction = self.interaction_projection(interaction)
        projected_social = self.social_projection(social)
        projected_interaction_proxy = self.interaction_projection(interaction_proxy)
        projected_social_proxy = self.social_projection(social_proxy)
        interaction_positive = projected_interaction @ projected_interaction_proxy
        interaction_negative = projected_interaction @ projected_social_proxy
        social_positive = projected_social @ projected_social_proxy
        social_negative = projected_social @ projected_interaction_proxy
        inter_view = (
            F.softplus(interaction_negative - interaction_positive).mean()
            + F.softplus(social_negative - social_positive).mean()
        )

        causal = torch.cat([output.causal_interaction, output.causal_social], dim=-1)
        bias = torch.cat([output.bias_interaction, output.bias_social], dim=-1)
        causal_center = causal.mean(dim=0)
        bias_center = bias.mean(dim=0)
        disagreement_terms = (
            (causal - causal_center).square().sum(dim=-1)
            - (causal - bias_center).square().sum(dim=-1)
            + margin
        )
        if hinged_disagreement:
            disagreement_terms = disagreement_terms.clamp_min(0.0)
        return inter_view, disagreement_terms.mean()

