from __future__ import annotations

import unittest

import numpy as np
import torch

from www2027.data import CascadeRecord, NextUserDataset, build_rolling_snapshots, make_temporal_environments
from www2027.metrics import (
    RankingAccumulator,
    aggregate_environment_metrics,
    popularity_group_ids,
    protected_union_scores,
    recency_group_ids,
)
from www2027.models import EnvironmentEncoder, TemporalDiffusionModel
from www2027.models.temporal_diffusion import DiffusionOutput
from www2027.training import (
    ERM,
    GroupDRO,
    VREx,
    environment_loss,
    prepare_environment,
    topk_subgroup_preservation_penalty,
)


def record(index: int, start: float, nodes: tuple[int, ...]) -> CascadeRecord:
    return CascadeRecord(
        cascade=nodes,
        timestamp=tuple(start + offset for offset in range(len(nodes))),
        source_split="synthetic",
        source_index=index,
    )


class TrainingComponentsTest(unittest.TestCase):
    def test_user_zero_is_not_used_as_padding(self) -> None:
        dataset = NextUserDataset(
            (record(0, 1, (0, 1, 2)),),
            num_nodes=3,
            max_prefix_length=4,
        )
        example = dataset[0]
        self.assertEqual(example["prefix"][0].item(), 0)
        self.assertEqual(example["prefix"][-1].item(), 3)

    def test_prepared_environment_and_model_forward(self) -> None:
        records = (
            record(0, 1, (0, 1, 2)),
            record(1, 2, (1, 2, 3)),
            record(2, 3, (2, 3, 4)),
            record(3, 4, (3, 4, 0)),
        )
        environments = make_temporal_environments(records, 2)
        snapshots = build_rolling_snapshots(environments, num_nodes=5)
        prepared = prepare_environment(snapshots[1], num_nodes=5, max_prefix_length=4)
        batch = prepared.dataset[0]
        model = TemporalDiffusionModel(
            num_nodes=5,
            dimension=8,
            rank=3,
            context_dim=4,
            environment_hidden_dim=8,
            dropout=0.0,
        )
        output = model(
            prefix=batch["prefix"].unsqueeze(0),
            elapsed=batch["elapsed"].unsqueeze(0),
            lengths=batch["length"].unsqueeze(0),
            edge_index=prepared.edge_index,
            edge_weight=prepared.edge_weight,
            environment_features=prepared.environment_features,
        )
        self.assertEqual(tuple(output.logits.shape), (1, 5))
        self.assertTrue(torch.isneginf(output.logits[0, batch["prefix"][0]]))
        breakdown = environment_loss(
            output,
            batch["target"].unsqueeze(0),
            prepared.local_popularity,
        )
        self.assertTrue(bool(torch.isfinite(breakdown.total)))
        self.assertLess(breakdown.prediction.item(), 15.0)
        breakdown.total.backward()

    def test_balanced_loss_uses_past_popularity(self) -> None:
        records = (
            record(0, 1, (0, 1, 2)),
            record(1, 2, (1, 2, 3)),
            record(2, 3, (2, 3, 4)),
            record(3, 4, (3, 4, 0)),
        )
        snapshots = build_rolling_snapshots(
            make_temporal_environments(records, 2), num_nodes=5
        )
        prepared = prepare_environment(snapshots[1], num_nodes=5, max_prefix_length=4)
        example = prepared.dataset[0]
        model = TemporalDiffusionModel(
            num_nodes=5,
            dimension=8,
            rank=3,
            context_dim=4,
            environment_hidden_dim=8,
            dropout=0.0,
        )
        output = model(
            example["prefix"].unsqueeze(0),
            example["elapsed"].unsqueeze(0),
            example["length"].unsqueeze(0),
            prepared.edge_index,
            prepared.edge_weight,
            prepared.environment_features,
        )
        balanced = environment_loss(
            output,
            example["target"].unsqueeze(0),
            prepared.local_popularity,
            prepared.historical_popularity,
            prepared.recent_popularity,
            popularity_balance_alpha=0.25,
            dormant_boost=0.5,
        )
        self.assertTrue(bool(torch.isfinite(balanced.total)))
        constrained = environment_loss(
            output,
            example["target"].unsqueeze(0),
            prepared.local_popularity,
            prepared.historical_popularity,
            prepared.recent_popularity,
            prepared.popularity_groups,
            prepared.recency_groups,
            constraint_weight=0.1,
            constraint_margin=0.5,
        )
        self.assertTrue(bool(torch.isfinite(constrained.total)))
        self.assertGreaterEqual(constrained.constraint_penalty.item(), 0.0)

    def test_group_dro_increases_weight_on_harder_environment(self) -> None:
        objective = GroupDRO(3, step_size=0.5)
        objective.train()
        combined = objective(
            torch.tensor([100.0, 1.0, 1.0], requires_grad=True),
            risk_for_update=torch.tensor([1.0, 3.0, 2.0]),
        )
        self.assertTrue(bool(torch.isfinite(combined)))
        self.assertGreater(objective.weights[1].item(), objective.weights[0].item())

    def test_erm_keeps_uniform_environment_weights(self) -> None:
        objective = ERM(3)
        losses = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
        combined = objective(losses)
        self.assertAlmostEqual(combined.item(), 2.0)
        self.assertTrue(torch.allclose(objective.weights, torch.full((3,), 1 / 3)))

    def test_vrex_penalizes_prediction_risk_variance(self) -> None:
        objective = VREx(3, penalty_weight=2.0)
        total_losses = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
        equal = objective(total_losses, torch.tensor([1.0, 1.0, 1.0]))
        unequal = objective(total_losses, torch.tensor([0.0, 1.0, 2.0]))
        self.assertAlmostEqual(equal.item(), 2.0)
        self.assertGreater(unequal.item(), equal.item())

    def test_static_mask_ignores_environment_features(self) -> None:
        model = TemporalDiffusionModel(
            num_nodes=5,
            dimension=8,
            rank=3,
            context_dim=4,
            environment_hidden_dim=8,
            dropout=0.0,
            mask_mode="static",
        )
        model.eval()
        prefix = torch.tensor([[0, 1, 5]])
        elapsed = torch.tensor([[0.0, 1.0, 0.0]])
        lengths = torch.tensor([2])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_weight = torch.ones(3)
        first = model(
            prefix,
            elapsed,
            lengths,
            edge_index,
            edge_weight,
            torch.zeros(EnvironmentEncoder.feature_dim),
        )
        second = model(
            prefix,
            elapsed,
            lengths,
            edge_index,
            edge_weight,
            torch.ones(EnvironmentEncoder.feature_dim),
        )
        self.assertTrue(torch.allclose(first.edge_mask, second.edge_mask))

    def test_no_mask_assigns_every_edge_to_stable_graph(self) -> None:
        model = TemporalDiffusionModel(
            num_nodes=5,
            dimension=8,
            rank=3,
            context_dim=4,
            environment_hidden_dim=8,
            dropout=0.0,
            mask_mode="none",
        )
        output = model(
            torch.tensor([[0, 5]]),
            torch.zeros(1, 2),
            torch.tensor([1]),
            torch.tensor([[0, 1], [1, 2]]),
            torch.ones(2),
            torch.zeros(EnvironmentEncoder.feature_dim),
        )
        self.assertTrue(torch.equal(output.edge_mask, torch.ones(2)))

    def test_temporal_prior_is_zero_initialized_and_uses_past_state(self) -> None:
        model = TemporalDiffusionModel(
            num_nodes=5,
            dimension=8,
            rank=3,
            context_dim=4,
            environment_hidden_dim=8,
            dropout=0.0,
            prior_mode="temporal",
        )
        inputs = {
            "prefix": torch.tensor([[0, 5]]),
            "elapsed": torch.zeros(1, 2),
            "lengths": torch.tensor([1]),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "edge_weight": torch.ones(2),
            "environment_features": torch.zeros(EnvironmentEncoder.feature_dim),
            "historical_popularity": torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0]),
            "recent_popularity": torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
        }
        output = model(**inputs)
        self.assertTrue(
            torch.equal(output.temporal_prior_logits, torch.zeros_like(output.logits))
        )
        output.logits[0, 1].backward()
        self.assertGreater(
            model.temporal_prior_gate[-1].weight.grad.abs().sum().item(),
            0.0,
        )

    def test_temporal_prior_rejects_missing_node_state(self) -> None:
        model = TemporalDiffusionModel(
            num_nodes=5,
            dimension=8,
            rank=3,
            context_dim=4,
            environment_hidden_dim=8,
            dropout=0.0,
            prior_mode="temporal",
        )
        with self.assertRaisesRegex(ValueError, "temporal prior requires"):
            model(
                torch.tensor([[0, 5]]),
                torch.zeros(1, 2),
                torch.tensor([1]),
                torch.tensor([[0, 1], [1, 2]]),
                torch.ones(2),
                torch.zeros(EnvironmentEncoder.feature_dim),
            )

    def test_topk_subgroup_anchor_penalizes_protected_candidate_drop(self) -> None:
        def output(logits: torch.Tensor, residual: torch.Tensor) -> DiffusionOutput:
            return DiffusionOutput(
                logits=logits,
                base_logits=logits - residual,
                shortcut_logits=torch.zeros_like(logits),
                temporal_prior_logits=residual,
                stable_representation=torch.zeros(1, 2),
                shortcut_representation=torch.zeros(1, 2),
                edge_mask=torch.ones(1),
            )

        groups = torch.tensor([0, 1, 0, 0])
        recency = torch.tensor([0, 1, 0, 0])
        unchanged = output(
            torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
            torch.zeros(1, 4),
        )
        self.assertEqual(
            topk_subgroup_preservation_penalty(
                unchanged, groups, recency, topk=2
            ).item(),
            0.0,
        )

        displaced = output(
            torch.tensor([[4.0, 0.0, 3.0, 2.0]], requires_grad=True),
            torch.tensor([[0.0, -3.0, 1.0, 1.0]]),
        )
        penalty = topk_subgroup_preservation_penalty(
            displaced, groups, recency, topk=2
        )
        self.assertGreater(penalty.item(), 0.0)
        penalty.backward()
        self.assertLess(displaced.logits.grad[0, 1].item(), 0.0)

    def test_protected_union_keeps_anchor_subgroup_candidate(self) -> None:
        anchor = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        adaptive = torch.tensor([[4.0, 0.0, 3.0, 2.0]])
        fused = protected_union_scores(
            adaptive,
            anchor,
            popularity_groups=torch.tensor([0, 1, 0, 0]),
            recency_groups=torch.tensor([0, 1, 0, 0]),
            topk=2,
        )
        fused_top2 = torch.topk(fused, 2, dim=1).indices.flatten().tolist()
        self.assertIn(1, fused_top2)
        self.assertIn(0, fused_top2)

    def test_protected_union_guarantees_nested_prefixes(self) -> None:
        anchor = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        adaptive = torch.tensor([[6.0, 2.0, 5.0, 4.0, 1.0, 3.0]])
        fused = protected_union_scores(
            adaptive,
            anchor,
            popularity_groups=torch.tensor([0, 1, 0, 0, 2, 0]),
            recency_groups=torch.tensor([0, 1, 0, 0, 1, 0]),
            topk=6,
            protected_cutoffs=(2, 5),
        )
        fused_top6 = torch.topk(fused, 6, dim=1).indices.flatten().tolist()
        self.assertIn(1, fused_top6[:2])
        self.assertIn(4, fused_top6[:5])
        self.assertEqual(len(set(fused_top6)), 6)

    def test_ranking_reports_mean_and_worst_environment(self) -> None:
        accumulator = RankingAccumulator((1, 2))
        accumulator.update(
            torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]),
            torch.tensor([0, 2]),
        )
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["hit@1"], 0.5)
        aggregate = aggregate_environment_metrics([metrics, {**metrics, "hit@1": 0.0}])
        self.assertAlmostEqual(aggregate["hit@1"], 0.25)
        self.assertAlmostEqual(aggregate["worst_hit@1"], 0.0)

    def test_past_only_user_strata(self) -> None:
        popularity_groups = popularity_group_ids(
            torch.tensor([10.0, 9.0, 5.0, 1.0, 0.0])
        )
        self.assertEqual(popularity_groups.tolist(), [0, 1, 2, 2, 3])
        recency_groups = recency_group_ids(
            torch.tensor([2.0, 1.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
        self.assertEqual(recency_groups.tolist(), [0, 1, 2])

    def test_ranking_accumulators_merge(self) -> None:
        first = RankingAccumulator((1,))
        second = RankingAccumulator((1,))
        first.update(torch.tensor([[2.0, 1.0]]), torch.tensor([0]))
        second.update(torch.tensor([[2.0, 1.0]]), torch.tensor([1]))
        first.merge(second)
        self.assertEqual(first.count, 2)
        self.assertAlmostEqual(first.compute()["hit@1"], 0.5)


if __name__ == "__main__":
    unittest.main()
