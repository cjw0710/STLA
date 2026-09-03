"""Environment-conditioned internal extension of the unchanged CIKM DeDiff.

The original one-layer graph path computes ``(A @ D) @ X`` and materializes an
``N x N`` product.  This module uses the algebraically equivalent
``A @ (D @ X)`` form and adds a zero-initialized, environment-conditioned
low-rank correction to ``D @ X``.  All original DeDiff parameters retain their
names, so a corrected CIKM checkpoint can be loaded directly.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataLoader import dataProcess  # type: ignore  # noqa: E402
from model import DeDiff  # type: ignore  # noqa: E402


class EnvironmentLowRankDebiasing(nn.Module):
    """Produce ``U diag(g(environment)) V^T X`` without an ``N x N`` tensor."""

    environment_feature_dim = 28

    def __init__(
        self,
        num_nodes: int,
        rank: int,
        *,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(num_nodes, rank, hidden_dim) < 1:
            raise ValueError("num_nodes, rank, and hidden_dim must be positive")
        self.num_nodes = num_nodes
        self.rank = rank
        self.left = nn.Parameter(torch.empty(num_nodes, rank))
        self.right = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.normal_(self.left, std=0.02)
        nn.init.normal_(self.right, std=0.02)
        self.environment_gate = nn.Sequential(
            nn.LayerNorm(self.environment_feature_dim),
            nn.Linear(self.environment_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.environment_gate[-1].weight)
        nn.init.zeros_(self.environment_gate[-1].bias)

    def coefficients(self, environment_features: torch.Tensor) -> torch.Tensor:
        if environment_features.shape[-1] != self.environment_feature_dim:
            raise ValueError("environment feature dimension must be 28")
        if environment_features.ndim != 1:
            raise ValueError("one environment feature vector is required")
        return torch.tanh(self.environment_gate(environment_features))

    def project_features(
        self,
        node_features: torch.Tensor,
        environment_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_features.ndim != 2 or node_features.shape[0] != self.num_nodes:
            raise ValueError("node_features must have shape [num_nodes, dim]")
        coefficients = self.coefficients(environment_features)
        compressed = self.right.transpose(0, 1) @ node_features
        correction = self.left @ (coefficients.unsqueeze(1) * compressed)
        return correction / math.sqrt(float(self.rank)), coefficients


class DynamicDeDiff(DeDiff):
    """DeDiff with an efficient environment-conditioned debiasing correction."""

    def __init__(
        self,
        args,
        *,
        temporal_rank: int = 8,
        temporal_hidden_dim: int = 32,
        temporal_dropout: float = 0.1,
    ) -> None:
        if args.gcn_layer != 1:
            raise ValueError("the associative DeDiff rewrite currently requires gcn_layer=1")
        super().__init__(args)
        self.temporal_debias = EnvironmentLowRankDebiasing(
            args.user_num,
            temporal_rank,
            hidden_dim=temporal_hidden_dim,
            dropout=temporal_dropout,
        )
        self.register_buffer("compressed_debias_left", None)
        self.register_buffer("compressed_debias_right", None)
        self.compressed_debias_rank = 0
        self.compressed_debias_energy = 1.0

    @staticmethod
    def _finish_one_layer_gcn(module: nn.Module, message: torch.Tensor) -> torch.Tensor:
        return module.batch_norm(message + module.GCN1.bias)

    @staticmethod
    def _graph_mm(graph: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if graph.layout in {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}:
            return torch.sparse.mm(graph, features)
        return graph @ features

    def temporal_parameters(self):
        return self.temporal_debias.parameters()

    @torch.no_grad()
    def compress_debiasing(self, rank: int) -> float:
        """Replace the dense static operator by its rank-``rank`` SVD factors."""

        if self.Debasing is None:
            raise RuntimeError("Debasing has already been compressed")
        maximum_rank = min(self.Debasing.shape)
        if rank < 1 or rank >= maximum_rank:
            raise ValueError(f"compression rank must be in [1, {maximum_rank})")
        left_vectors, singular_values, right_vectors = torch.linalg.svd(
            self.Debasing.detach(),
            full_matrices=False,
        )
        retained = singular_values[:rank]
        scale = torch.sqrt(retained)
        self.compressed_debias_left = left_vectors[:, :rank] * scale.unsqueeze(0)
        self.compressed_debias_right = right_vectors[:rank, :].transpose(0, 1) * scale.unsqueeze(0)
        energy = retained.square().sum() / singular_values.square().sum().clamp_min(1e-12)
        self.Debasing = None
        self.compressed_debias_rank = rank
        self.compressed_debias_energy = float(energy)
        return self.compressed_debias_energy

    def base_project_features(self, node_features: torch.Tensor) -> torch.Tensor:
        if self.Debasing is not None:
            return self.Debasing @ node_features
        if self.compressed_debias_left is None or self.compressed_debias_right is None:
            raise RuntimeError("compressed debiasing factors are unavailable")
        return self.compressed_debias_left @ (
            self.compressed_debias_right.transpose(0, 1) @ node_features
        )

    def freeze_anchor(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.temporal_parameters():
            parameter.requires_grad_(True)

    def forward(self, args, data, info):
        cascade, cas_mask, label, label_mask, _, _, dis, timestamp = dataProcess(args, data)
        interaction_graph = info.get("A_interaction_dynamic", info["A_interaction"])
        social_graph = info.get("A_social_dynamic", info["A_social"])
        environment_features = info.get("environment_features")
        if environment_features is None:
            raise ValueError("DynamicDeDiff requires past-only environment_features")

        base_projected = self.base_project_features(self.UEm.weight)
        temporal_projected, coefficients = self.temporal_debias.project_features(
            self.UEm.weight,
            environment_features,
        )
        causal_projected = base_projected + temporal_projected

        causal_interaction_message = self._graph_mm(interaction_graph, causal_projected)
        causal_social_message = self._graph_mm(social_graph, causal_projected)
        full_interaction_message = self._graph_mm(interaction_graph, self.UEm.weight)
        full_social_message = self._graph_mm(social_graph, self.UEm.weight)
        bias_interaction_message = full_interaction_message - causal_interaction_message
        bias_social_message = full_social_message - causal_social_message

        h_casual_interaction = self._finish_one_layer_gcn(
            self.GCN1,
            causal_interaction_message,
        )
        h_bias_interaction = self._finish_one_layer_gcn(
            self.GCN2,
            bias_interaction_message,
        )
        h_casual_social = self._finish_one_layer_gcn(
            self.GCN3,
            causal_social_message,
        )
        h_bias_social = self._finish_one_layer_gcn(
            self.GCN4,
            bias_social_message,
        )

        process_data = {
            "embedding_isc": h_casual_interaction + h_casual_social,
            "embedding_isb": h_bias_interaction + h_bias_social,
            "temporal_coefficients": coefficients,
        }
        embedding_tc_proxy = h_casual_interaction.mean(dim=0)
        embedding_sc_proxy_sum = self._graph_mm(
            social_graph.transpose(0, 1),
            h_casual_social,
        )
        embedding_sc_proxy = embedding_sc_proxy_sum.mean(dim=0)
        process_data["e_T_prime"] = self.mlp_t(h_casual_interaction)
        process_data["e_S_prime"] = self.mlp_s(h_casual_social)
        process_data["p_T_prime"] = self.mlp_t(embedding_tc_proxy)
        process_data["p_S_prime"] = self.mlp_s(embedding_sc_proxy)

        user_embed1 = F.embedding(cascade, h_casual_interaction, padding_idx=0)
        user_embed2 = F.embedding(cascade, h_casual_social, padding_idx=0)
        cascade_embedding = self.Fusion2(user_embed1, user_embed2)
        temporal_hidden = self.TEAN(cascade_embedding, cas_mask, timestamp)
        social_hidden = self.SSAN(cascade_embedding, cas_mask, dis)
        hidden = self.Fusion(temporal_hidden, social_hidden)

        user_embed3 = F.embedding(cascade, h_bias_interaction, padding_idx=0)
        user_embed4 = F.embedding(cascade, h_bias_social, padding_idx=0)
        bias_cascade_embedding = self.Fusion2(user_embed3, user_embed4)
        bias_temporal_hidden = self.TEAN(bias_cascade_embedding, cas_mask, timestamp)
        bias_social_hidden = self.SSAN(bias_cascade_embedding, cas_mask, dis)
        process_data["h_b"] = self.Fusion(bias_temporal_hidden, bias_social_hidden)
        process_data["frequency"] = info["frequency"]

        prediction = self.Predict(hidden) + label_mask
        return prediction.view(-1, prediction.size(-1)), label, process_data
