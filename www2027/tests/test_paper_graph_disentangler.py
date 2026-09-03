from __future__ import annotations

import unittest

import torch

from www2027.models import PaperGraphDisentangler, PaperLowRankEdgeMask


class PaperLowRankEdgeMaskTest(unittest.TestCase):
    def test_sparse_scores_match_dense_equation_two_reference(self) -> None:
        module = PaperLowRankEdgeMask(num_nodes=5, rank=3)
        with torch.no_grad():
            module.left.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)
            module.right.copy_(torch.arange(15, 30, dtype=torch.float32).reshape(5, 3) / 10)
        edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 0]], dtype=torch.long)

        sparse = module(edges)
        dense_reference = torch.sigmoid(module.left @ module.right.transpose(0, 1))
        self.assertTrue(torch.allclose(sparse, dense_reference[edges[0], edges[1]]))

    def test_split_is_complementary_and_keeps_only_observed_edges(self) -> None:
        module = PaperLowRankEdgeMask(num_nodes=100, rank=8)
        edges = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)
        weights = torch.tensor([1.0, 2.0, 4.0])
        split = module.split(edges, weights)

        self.assertEqual(tuple(split.mask.shape), (3,))
        self.assertTrue(torch.allclose(split.causal_weight + split.bias_weight, weights))
        self.assertTrue(bool(torch.all((split.mask >= 0) & (split.mask <= 1))))
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 2 * 100 * 8)
        self.assertEqual(tuple(module.left.shape), (100, 8))
        self.assertEqual(tuple(module.right.shape), (100, 8))

    def test_one_shared_mask_is_reused_across_graph_views(self) -> None:
        model = PaperGraphDisentangler(num_nodes=6, dimension=4, rank=2)
        common_edge = torch.tensor([[1], [4]], dtype=torch.long)
        interaction = torch.cat([common_edge, torch.tensor([[0], [2]])], dim=1)
        social = torch.cat([torch.tensor([[3], [5]]), common_edge], dim=1)
        output = model(
            interaction,
            torch.ones(2),
            social,
            torch.ones(2),
        )
        self.assertTrue(
            torch.allclose(output.interaction_split.mask[0], output.social_split.mask[1])
        )


class PaperGraphDisentanglerTest(unittest.TestCase):
    def test_four_sparse_views_and_losses_are_differentiable(self) -> None:
        torch.manual_seed(11)
        model = PaperGraphDisentangler(
            num_nodes=7,
            dimension=5,
            rank=3,
            gcn_layers=2,
        )
        interaction_edges = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long
        )
        social_edges = torch.tensor(
            [[0, 2, 4, 6, 1], [2, 4, 6, 1, 3]], dtype=torch.long
        )
        output = model(
            interaction_edges,
            torch.ones(interaction_edges.shape[1]),
            social_edges,
            torch.ones(social_edges.shape[1]),
        )

        for embedding in (
            output.causal_interaction,
            output.bias_interaction,
            output.causal_social,
            output.bias_social,
        ):
            self.assertEqual(tuple(embedding.shape), (7, 5))
            self.assertTrue(bool(torch.isfinite(embedding).all()))

        inter_view, disagreement = model.disentanglement_losses(output)
        corrected_inter_view, hinged = model.disentanglement_losses(
            output, hinged_disagreement=True
        )
        self.assertTrue(torch.allclose(inter_view, corrected_inter_view))
        self.assertTrue(bool(torch.isfinite(inter_view)))
        self.assertTrue(bool(torch.isfinite(disagreement)))
        self.assertGreaterEqual(hinged.item(), 0.0)
        (inter_view + disagreement).backward()
        self.assertIsNotNone(model.edge_mask.left.grad)
        self.assertTrue(bool(torch.isfinite(model.edge_mask.left.grad).all()))

    def test_empty_observed_graph_still_propagates_self_loops(self) -> None:
        model = PaperGraphDisentangler(num_nodes=4, dimension=3, rank=2)
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_weights = torch.empty(0)
        output = model(
            empty_edges,
            empty_weights,
            empty_edges,
            empty_weights,
        )
        self.assertEqual(tuple(output.causal_interaction.shape), (4, 3))
        self.assertTrue(bool(torch.isfinite(output.causal_interaction).all()))


if __name__ == "__main__":
    unittest.main()
