from __future__ import annotations

import unittest

import torch

from www2027.models import TemporalLogitAdapter


class TemporalLogitAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = TemporalLogitAdapter(
            6,
            context_dim=4,
            hidden_dim=8,
            environment_hidden_dim=8,
            dropout=0.0,
        )
        self.anchor = torch.randn(6, 6)
        self.sequence = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
        self.timestamps = torch.tensor([[1.0, 2.0, 3.0, -1.0], [2.0, 3.0, 5.0, 8.0]])
        self.environment = torch.randn(28)
        self.historical = torch.tensor([3.0, 2.0, 1.0, 0.0, 4.0, 0.0])
        self.recent = torch.tensor([0.0, 2.0, 0.0, 0.0, 1.0, 0.0])

    def test_zero_initialization_exactly_reproduces_anchor(self) -> None:
        adapted, residual = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
        )
        torch.testing.assert_close(adapted, self.anchor, rtol=0.0, atol=0.0)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)

    def test_gradients_reach_gate_and_environment_encoder(self) -> None:
        adapted, _ = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
        )
        torch.nn.functional.cross_entropy(adapted, torch.tensor([0, 1, 2, 3, 4, 5])).backward()
        self.assertIsNotNone(self.adapter.gate[-1].weight.grad)
        self.assertGreater(float(self.adapter.gate[-1].weight.grad.abs().sum()), 0.0)

    def test_prefix_features_are_causal(self) -> None:
        original = self.adapter.prefix_features(
            self.sequence,
            self.timestamps,
            self.historical,
            self.recent,
        )
        changed = self.sequence.clone()
        changed[0, 2] = 7
        perturbed = self.adapter.prefix_features(
            changed,
            self.timestamps,
            self.historical,
            self.recent,
        )
        torch.testing.assert_close(original[0, :2], perturbed[0, :2])

    def test_zero_offset_supports_original_dediff_user_ids(self) -> None:
        sequence = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 1]])
        adapted, residual = self.adapter(
            self.anchor,
            sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
            input_id_offset=0,
        )
        torch.testing.assert_close(adapted, self.anchor, rtol=0.0, atol=0.0)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)

    def test_low_rank_node_codes_keep_exact_initial_fallback(self) -> None:
        adapter = TemporalLogitAdapter(
            6,
            context_dim=4,
            hidden_dim=8,
            environment_hidden_dim=8,
            dropout=0.0,
            node_rank=3,
        )
        adapted, residual = adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
        )
        self.assertEqual(adapter.temporal_node_features(self.historical, self.recent).shape, (6, 8))
        torch.testing.assert_close(adapted, self.anchor, rtol=0.0, atol=0.0)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)

    def test_default_matches_explicit_full_ablation(self) -> None:
        with torch.no_grad():
            self.adapter.gate[-1].weight.normal_()
            self.adapter.gate[-1].bias.normal_()
        default = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
        )
        explicit = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
            ablation="full",
        )
        torch.testing.assert_close(default[0], explicit[0])
        torch.testing.assert_close(default[1], explicit[1])

    def test_no_environment_ignores_environment_features(self) -> None:
        with torch.no_grad():
            self.adapter.gate[-1].weight.normal_()
            self.adapter.gate[-1].bias.normal_()
        first = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
            ablation="no_environment",
        )[0]
        second = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment + 100.0,
            self.historical,
            self.recent,
            ablation="no_environment",
        )[0]
        torch.testing.assert_close(first, second)

    def test_no_prefix_ignores_sequence_and_timestamp_descriptors(self) -> None:
        with torch.no_grad():
            self.adapter.gate[-1].weight.normal_()
            self.adapter.gate[-1].bias.normal_()
        first = self.adapter(
            self.anchor,
            self.sequence,
            self.timestamps,
            self.environment,
            self.historical,
            self.recent,
            ablation="no_prefix",
        )[0]
        changed_sequence = torch.where(self.sequence.ne(0), torch.full_like(self.sequence, 7), 0)
        second = self.adapter(
            self.anchor,
            changed_sequence,
            self.timestamps + 50.0,
            self.environment,
            self.historical,
            self.recent,
            ablation="no_prefix",
        )[0]
        torch.testing.assert_close(first, second)

    def test_historical_only_masks_all_other_node_features(self) -> None:
        features = self.adapter.temporal_node_features(
            self.historical,
            self.recent,
            ablation="historical_only",
        )
        self.assertGreater(float(features[:, 0].abs().sum()), 0.0)
        torch.testing.assert_close(features[:, 1:], torch.zeros_like(features[:, 1:]))

    def test_unknown_ablation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown adapter ablation"):
            self.adapter.temporal_node_features(
                self.historical,
                self.recent,
                ablation="not-a-real-ablation",
            )


if __name__ == "__main__":
    unittest.main()
