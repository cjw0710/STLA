from __future__ import annotations

import unittest

import torch

from www2027.models.dynamic_dediff import EnvironmentLowRankDebiasing


class EnvironmentLowRankDebiasingTest(unittest.TestCase):
    def test_zero_gate_produces_exact_zero_correction(self) -> None:
        module = EnvironmentLowRankDebiasing(7, 3, hidden_dim=8, dropout=0.0)
        features = torch.randn(7, 5)
        correction, coefficients = module.project_features(features, torch.randn(28))
        torch.testing.assert_close(coefficients, torch.zeros_like(coefficients), rtol=0.0, atol=0.0)
        torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0.0, atol=0.0)

    def test_gradient_reaches_environment_gate(self) -> None:
        module = EnvironmentLowRankDebiasing(7, 3, hidden_dim=8, dropout=0.0)
        correction, _ = module.project_features(torch.randn(7, 5), torch.randn(28))
        correction.square().sum().backward()
        # A squared zero correction has zero derivative; use a linear probe to
        # verify the zero-initialized gate is trainable.
        module.zero_grad(set_to_none=True)
        correction, _ = module.project_features(torch.randn(7, 5), torch.randn(28))
        correction.sum().backward()
        gradient = module.environment_gate[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_associative_graph_rewrite_matches_dense_product(self) -> None:
        graph = torch.randn(9, 9)
        debiasing = torch.randn(9, 9)
        features = torch.randn(9, 4)
        dense = (graph @ debiasing) @ features
        efficient = graph @ (debiasing @ features)
        torch.testing.assert_close(efficient, dense, rtol=1e-5, atol=1e-5)

    def test_sparse_graph_product_matches_dense(self) -> None:
        graph = torch.randn(9, 9)
        graph[graph.abs() < 0.8] = 0.0
        features = torch.randn(9, 4)
        dense = graph @ features
        sparse = torch.sparse.mm(graph.to_sparse_coo(), features)
        torch.testing.assert_close(sparse, dense)

    def test_svd_factor_projection_matches_truncated_matrix(self) -> None:
        matrix = torch.randn(11, 11)
        features = torch.randn(11, 5)
        left_vectors, singular_values, right_vectors = torch.linalg.svd(matrix)
        rank = 4
        scale = torch.sqrt(singular_values[:rank])
        left = left_vectors[:, :rank] * scale
        right = right_vectors[:rank, :].transpose(0, 1) * scale
        expected = (
            left_vectors[:, :rank]
            @ torch.diag(singular_values[:rank])
            @ right_vectors[:rank, :]
            @ features
        )
        factored = left @ (right.transpose(0, 1) @ features)
        torch.testing.assert_close(factored, expected)


if __name__ == "__main__":
    unittest.main()
