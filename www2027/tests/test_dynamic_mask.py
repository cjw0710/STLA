from __future__ import annotations

import unittest

import torch

from www2027.models import DynamicLowRankMask, EnvironmentEncoder, build_environment_features


class DynamicLowRankMaskTest(unittest.TestCase):
    def test_initial_mask_has_edge_and_environment_variation(self) -> None:
        torch.manual_seed(7)
        module = DynamicLowRankMask(num_nodes=50, rank=8, context_dim=6)
        edge_index = torch.stack(
            [torch.arange(40, dtype=torch.long), torch.arange(1, 41, dtype=torch.long)]
        )
        first = module(edge_index, torch.zeros(6))
        second = module(edge_index, torch.ones(6))
        self.assertGreater(first.std(unbiased=False).item(), 0.005)
        self.assertGreater(torch.mean(torch.abs(first - second)).item(), 1e-4)

    def test_mask_is_edge_sparse_and_weight_split_is_exact(self) -> None:
        module = DynamicLowRankMask(num_nodes=100, rank=8, context_dim=6)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
        context = torch.randn(6)

        stable, shortcut, mask = module.split_weights(edge_index, edge_weight, context)
        self.assertEqual(tuple(mask.shape), (4,))
        self.assertTrue(bool(torch.all((mask >= 0) & (mask <= 1))))
        self.assertTrue(torch.allclose(stable + shortcut, edge_weight))

    def test_parameter_growth_is_linear_in_nodes(self) -> None:
        small = DynamicLowRankMask(num_nodes=100, rank=4, context_dim=3)
        large = DynamicLowRankMask(num_nodes=200, rank=4, context_dim=3)
        extra_parameters = sum(p.numel() for p in large.parameters()) - sum(
            p.numel() for p in small.parameters()
        )
        self.assertEqual(extra_parameters, 2 * 100 * 4)

    def test_gradients_reach_node_factors_and_context_layers(self) -> None:
        module = DynamicLowRankMask(num_nodes=6, rank=3, context_dim=4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        loss = module(edge_index, torch.randn(4)).sum()
        loss.backward()
        self.assertIsNotNone(module.left.weight.grad)
        self.assertIsNotNone(module.left_scale.weight.grad)
        self.assertTrue(bool(torch.isfinite(module.left.weight.grad).all()))

    def test_environment_features_and_encoder(self) -> None:
        features = build_environment_features(
            popularity=torch.tensor([0.0, 2.0, 4.0, 1.0]),
            degree=torch.tensor([1.0, 3.0, 2.0, 1.0]),
        )
        self.assertEqual(tuple(features.shape), (EnvironmentEncoder.feature_dim,))
        context = EnvironmentEncoder(context_dim=5)(features)
        self.assertEqual(tuple(context.shape), (5,))
        self.assertTrue(bool(torch.isfinite(context).all()))


if __name__ == "__main__":
    unittest.main()
