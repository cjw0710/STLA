from __future__ import annotations

import unittest

import numpy as np

from www2027.data import (
    CascadeRecord,
    build_interaction_graph,
    build_rolling_snapshots,
    chronological_split,
    make_temporal_environments,
    popularity_counts,
)


def record(index: int, start: float, cascade: tuple[int, ...]) -> CascadeRecord:
    return CascadeRecord(
        cascade=cascade,
        timestamp=tuple(start + offset for offset in range(len(cascade))),
        source_split="synthetic",
        source_index=index,
    )


class TemporalPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            record(0, 1, (0, 1, 2)),
            record(1, 2, (1, 2)),
            record(2, 2, (2, 3)),
            record(3, 3, (3, 4)),
            record(4, 4, (4, 0)),
            record(5, 5, (0, 2)),
            record(6, 6, (2, 4)),
        )

    def test_split_is_chronological_and_keeps_ties_together(self) -> None:
        split = chronological_split(self.records, ratios=(0.5, 0.2, 0.3))
        partitions = (split.train, split.valid, split.test)
        self.assertTrue(all(partition for partition in partitions))
        self.assertLess(split.train[-1].start_time, split.valid[0].start_time)
        self.assertLess(split.valid[-1].start_time, split.test[0].start_time)

        timestamp_two_partitions = sum(
            any(item.start_time == 2 for item in partition) for partition in partitions
        )
        self.assertEqual(timestamp_two_partitions, 1)

    def test_rolling_snapshot_excludes_current_and_future_environments(self) -> None:
        environments = make_temporal_environments(self.records, 3)
        snapshots = build_rolling_snapshots(environments, num_nodes=5, sample_hop=1)

        self.assertEqual(snapshots[0].history_size, 0)
        np.testing.assert_array_equal(snapshots[0].popularity, np.zeros(5))
        self.assertEqual(snapshots[1].history_size, len(environments[0].records))
        np.testing.assert_array_equal(
            snapshots[1].popularity,
            popularity_counts(environments[0].records, num_nodes=5),
        )
        self.assertEqual(snapshots[1].recent_history_size, len(environments[0].records))
        np.testing.assert_array_equal(
            snapshots[1].recent_popularity,
            popularity_counts(environments[0].records, num_nodes=5),
        )
        if len(snapshots) > 2:
            np.testing.assert_array_equal(
                snapshots[2].recent_popularity,
                popularity_counts(environments[1].records, num_nodes=5),
            )

    def test_interaction_graph_accumulates_observed_edges(self) -> None:
        graph = build_interaction_graph(
            (record(0, 1, (0, 1, 2)),),
            num_nodes=3,
            sample_hop=1,
            undirected=False,
            add_self_loops=False,
        )
        self.assertEqual(graph.shape, (3, 3))
        self.assertEqual(graph[0, 1], 1.0)
        self.assertEqual(graph[1, 2], 1.0)
        self.assertEqual(graph[0, 2], 0.0)

    def test_invalid_intra_cascade_time_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CascadeRecord((0, 1), (2.0, 1.0), "synthetic", 0)


if __name__ == "__main__":
    unittest.main()
