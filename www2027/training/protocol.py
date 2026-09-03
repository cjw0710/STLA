"""Convert leakage-safe temporal snapshots into model-ready environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data import NextUserDataset, RollingSnapshot, popularity_counts
from ..metrics import popularity_group_ids, recency_group_ids
from ..models import build_environment_features


@dataclass
class PreparedEnvironment:
    name: str
    dataset: NextUserDataset
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    environment_features: torch.Tensor
    historical_popularity: torch.Tensor
    recent_popularity: torch.Tensor
    popularity_groups: torch.Tensor
    recency_groups: torch.Tensor
    local_popularity: torch.Tensor

    def graph_to(self, device: torch.device) -> "PreparedEnvironment":
        return PreparedEnvironment(
            name=self.name,
            dataset=self.dataset,
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            environment_features=self.environment_features.to(device),
            historical_popularity=self.historical_popularity.to(device),
            recent_popularity=self.recent_popularity.to(device),
            popularity_groups=self.popularity_groups.to(device),
            recency_groups=self.recency_groups.to(device),
            local_popularity=self.local_popularity.to(device),
        )


def prepare_environment(
    snapshot: RollingSnapshot,
    num_nodes: int,
    max_prefix_length: int,
) -> PreparedEnvironment:
    """Prepare sparse graph tensors and past-only context features."""

    graph = snapshot.interaction_graph.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([graph.row, graph.col]).astype(np.int64, copy=False)
    )
    edge_weight = torch.from_numpy(graph.data.astype(np.float32, copy=False))
    degree = np.asarray(snapshot.interaction_graph.sum(axis=1)).reshape(-1)
    recent_degree = np.asarray(snapshot.recent_interaction_graph.sum(axis=1)).reshape(-1)
    features = build_environment_features(
        torch.from_numpy(snapshot.popularity.astype(np.float32, copy=False)),
        torch.from_numpy(degree.astype(np.float32, copy=False)),
        torch.from_numpy(snapshot.recent_popularity.astype(np.float32, copy=False)),
        torch.from_numpy(recent_degree.astype(np.float32, copy=False)),
    )
    local_popularity = popularity_counts(snapshot.environment.records, num_nodes)
    historical_popularity = torch.from_numpy(
        snapshot.popularity.astype(np.float32, copy=False)
    )
    recent_popularity = torch.from_numpy(
        snapshot.recent_popularity.astype(np.float32, copy=False)
    )
    return PreparedEnvironment(
        name=snapshot.environment.name,
        dataset=NextUserDataset(
            snapshot.environment.records,
            num_nodes=num_nodes,
            max_prefix_length=max_prefix_length,
        ),
        edge_index=edge_index,
        edge_weight=edge_weight,
        environment_features=features,
        historical_popularity=historical_popularity,
        recent_popularity=recent_popularity,
        popularity_groups=popularity_group_ids(historical_popularity),
        recency_groups=recency_group_ids(historical_popularity, recent_popularity),
        local_popularity=torch.from_numpy(local_popularity.astype(np.float32, copy=False)),
    )
