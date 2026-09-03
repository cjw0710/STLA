"""Past-only graph construction for sequential temporal environments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .temporal_split import CascadeRecord, TemporalEnvironment


@dataclass(frozen=True)
class RollingSnapshot:
    """Graph and popularity statistics available before one environment."""

    environment: TemporalEnvironment
    history_size: int
    interaction_graph: sparse.csr_matrix
    popularity: np.ndarray
    recent_history_size: int
    recent_interaction_graph: sparse.csr_matrix
    recent_popularity: np.ndarray


def _validate_node(node: int, num_nodes: int) -> None:
    if node < 0 or node >= num_nodes:
        raise ValueError(f"node id {node} is outside [0, {num_nodes})")


def popularity_counts(
    records: Iterable[CascadeRecord],
    num_nodes: int,
) -> np.ndarray:
    """Count user activations in a collection of cascades."""

    counts = np.zeros(num_nodes, dtype=np.float64)
    for record in records:
        for node in record.cascade:
            _validate_node(node, num_nodes)
            counts[node] += 1.0
    return counts


def build_interaction_graph(
    records: Iterable[CascadeRecord],
    num_nodes: int,
    sample_hop: int = 2,
    *,
    undirected: bool = True,
    add_self_loops: bool = True,
    binary: bool = False,
) -> sparse.csr_matrix:
    """Construct a sparse cascade-interaction graph from historical data only.

    Each activated node connects to up to ``sample_hop`` preceding activations
    in the same cascade. Repeated observations become edge weights unless
    ``binary`` is requested.
    """

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if sample_hop < 1:
        raise ValueError("sample_hop must be positive")

    weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    for record in records:
        for node in record.cascade:
            _validate_node(node, num_nodes)
        for destination_index in range(1, len(record.cascade)):
            destination = record.cascade[destination_index]
            source_start = max(0, destination_index - sample_hop)
            for source_index in range(source_start, destination_index):
                source = record.cascade[source_index]
                weights[(source, destination)] += 1.0
                if undirected and source != destination:
                    weights[(destination, source)] += 1.0

    if add_self_loops:
        for node in range(num_nodes):
            weights[(node, node)] += 1.0

    if not weights:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    rows = np.fromiter((edge[0] for edge in weights), dtype=np.int64)
    columns = np.fromiter((edge[1] for edge in weights), dtype=np.int64)
    values = np.fromiter(weights.values(), dtype=np.float32)
    if binary:
        values.fill(1.0)
    return sparse.csr_matrix((values, (rows, columns)), shape=(num_nodes, num_nodes))


def build_rolling_snapshots(
    environments: Sequence[TemporalEnvironment],
    num_nodes: int,
    *,
    warm_start_records: Iterable[CascadeRecord] = (),
    warm_start_recent_records: Iterable[CascadeRecord] = (),
    sample_hop: int = 2,
    undirected: bool = True,
    add_self_loops: bool = True,
    binary: bool = False,
) -> tuple[RollingSnapshot, ...]:
    """Build one snapshot per environment using only preceding cascades."""

    history = list(warm_start_records)
    recent_history = list(warm_start_recent_records)
    if recent_history and not history:
        raise ValueError("recent warm-start records require cumulative warm-start history")
    history_keys = {
        (record.source_split, record.source_index, record.start_time) for record in history
    }
    if any(
        (record.source_split, record.source_index, record.start_time) not in history_keys
        for record in recent_history
    ):
        raise ValueError("recent warm-start records must be a subset of warm-start history")
    snapshots: list[RollingSnapshot] = []
    previous_end: float | None = None
    for environment in environments:
        if previous_end is not None and environment.start_time < previous_end:
            raise ValueError("environments must be supplied in chronological order")
        if history and max(record.start_time for record in history) >= environment.start_time:
            raise ValueError("history must end strictly before the environment begins")

        snapshots.append(
            RollingSnapshot(
                environment=environment,
                history_size=len(history),
                interaction_graph=build_interaction_graph(
                    history,
                    num_nodes,
                    sample_hop,
                    undirected=undirected,
                    add_self_loops=add_self_loops,
                    binary=binary,
                ),
                popularity=popularity_counts(history, num_nodes),
                recent_history_size=len(recent_history),
                recent_interaction_graph=build_interaction_graph(
                    recent_history,
                    num_nodes,
                    sample_hop,
                    undirected=undirected,
                    add_self_loops=add_self_loops,
                    binary=binary,
                ),
                recent_popularity=popularity_counts(recent_history, num_nodes),
            )
        )
        history.extend(environment.records)
        recent_history = list(environment.records)
        previous_end = environment.end_time

    return tuple(snapshots)
