from __future__ import annotations

import unittest

import torch

from www2027.models import PaperFaithfulDeDiff, PaperSTAN, paper_dediff_loss


class PaperSTANTest(unittest.TestCase):
    def test_future_tokens_cannot_change_past_hidden_states(self) -> None:
        torch.manual_seed(3)
        module = PaperSTAN(
            dimension=6,
            heads=3,
            head_dimension=4,
            social_distance_bins=8,
            time_interval_bins=8,
            prior_dimension=4,
            dropout=0.0,
        )
        module.eval()
        first = torch.randn(1, 4, 6)
        second = first.clone()
        second[:, 2:] = torch.randn_like(second[:, 2:]) * 20.0
        elapsed = torch.tensor([[0.0, 1.0, 2.0, 4.0]])
        distance = torch.zeros(1, 4, 4)
        lengths = torch.tensor([4])

        first_hidden = module(first, elapsed, distance, lengths)
        second_hidden = module(second, elapsed, distance, lengths)
        self.assertTrue(torch.allclose(first_hidden[:, :2], second_hidden[:, :2], atol=1e-6))
        self.assertFalse(torch.allclose(first_hidden[:, 2:], second_hidden[:, 2:]))

    def test_padding_queries_are_zero_and_ignored(self) -> None:
        torch.manual_seed(5)
        module = PaperSTAN(
            dimension=4,
            heads=2,
            head_dimension=4,
            dropout=0.0,
        )
        module.eval()
        sequence = torch.randn(2, 5, 4)
        elapsed = torch.zeros(2, 5)
        distance = torch.zeros(2, 5, 5)
        lengths = torch.tensor([2, 4])
        hidden = module(sequence, elapsed, distance, lengths)
        self.assertTrue(torch.equal(hidden[0, 2:], torch.zeros_like(hidden[0, 2:])))
        self.assertTrue(torch.equal(hidden[1, 4:], torch.zeros_like(hidden[1, 4:])))


class PaperFaithfulDeDiffTest(unittest.TestCase):
    @staticmethod
    def graph_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interaction = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
        )
        social = torch.tensor(
            [[0, 2, 4, 1], [2, 4, 1, 3]], dtype=torch.long
        )
        return interaction, torch.ones(5), social, torch.ones(4)

    def test_forward_masks_seen_real_users_and_excludes_padding_candidate(self) -> None:
        torch.manual_seed(7)
        model = PaperFaithfulDeDiff(
            num_nodes=5,
            dimension=8,
            rank=3,
            attention_heads=2,
            attention_head_dimension=4,
            dropout=0.0,
        )
        interaction, interaction_weight, social, social_weight = self.graph_inputs()
        output = model(
            prefix=torch.tensor([[0, 1, 5], [2, 5, 5]]),
            elapsed=torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
            lengths=torch.tensor([2, 1]),
            social_distance=torch.zeros(2, 3, 3),
            interaction_edge_index=interaction,
            interaction_edge_weight=interaction_weight,
            social_edge_index=social,
            social_edge_weight=social_weight,
        )
        self.assertEqual(tuple(output.logits.shape), (2, 5))
        self.assertTrue(torch.isneginf(output.logits[0, 0]))
        self.assertTrue(torch.isneginf(output.logits[0, 1]))
        self.assertTrue(torch.isneginf(output.logits[1, 2]))
        self.assertEqual(output.bias_logits.shape, output.logits.shape)

    def test_composite_loss_is_finite_and_reaches_graph_and_stan(self) -> None:
        torch.manual_seed(13)
        model = PaperFaithfulDeDiff(
            num_nodes=5,
            dimension=8,
            rank=3,
            attention_heads=2,
            attention_head_dimension=4,
            dropout=0.0,
        )
        interaction, interaction_weight, social, social_weight = self.graph_inputs()
        output = model(
            prefix=torch.tensor([[0, 1, 5], [1, 3, 5]]),
            elapsed=torch.tensor([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            lengths=torch.tensor([2, 2]),
            social_distance=torch.tensor(
                [
                    [[0.0, 1.0, 31.0], [1.0, 0.0, 31.0], [31.0, 31.0, 31.0]],
                    [[0.0, 2.0, 31.0], [2.0, 0.0, 31.0], [31.0, 31.0, 31.0]],
                ]
            ),
            interaction_edge_index=interaction,
            interaction_edge_weight=interaction_weight,
            social_edge_index=social,
            social_edge_weight=social_weight,
        )
        loss = paper_dediff_loss(
            output,
            target=torch.tensor([2, 4]),
            popularity_target=torch.tensor([8.0, 4.0, 2.0, 1.0, 0.0]),
            alpha=0.1,
            lambda_disagreement=0.01,
            lambda_inter_view=0.01,
        )
        self.assertTrue(bool(torch.isfinite(loss.total)))
        self.assertTrue(bool(torch.isfinite(loss.prediction)))
        self.assertTrue(bool(torch.isfinite(loss.bias)))
        loss.total.backward()
        self.assertIsNotNone(model.graph_disentangler.edge_mask.left.grad)
        self.assertIsNotNone(model.causal_stan.query.weight.grad)
        self.assertIsNotNone(model.bias_stan.query.weight.grad)
        self.assertGreater(model.graph_disentangler.edge_mask.left.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.causal_stan.query.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.bias_stan.query.weight.grad.abs().sum().item(), 0.0)

    def test_both_documented_kl_directions_are_explicit(self) -> None:
        model = PaperFaithfulDeDiff(
            num_nodes=5,
            dimension=4,
            rank=2,
            attention_heads=1,
            attention_head_dimension=4,
            dropout=0.0,
        )
        interaction, interaction_weight, social, social_weight = self.graph_inputs()
        output = model(
            torch.tensor([[0, 5]]),
            torch.zeros(1, 2),
            torch.tensor([1]),
            torch.zeros(1, 2, 2),
            interaction,
            interaction_weight,
            social,
            social_weight,
        )
        popularity = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        forward_kl = paper_dediff_loss(
            output,
            torch.tensor([1]),
            popularity,
            kl_direction="prediction_to_target",
        )
        reverse_kl = paper_dediff_loss(
            output,
            torch.tensor([1]),
            popularity,
            kl_direction="target_to_prediction",
        )
        self.assertTrue(bool(torch.isfinite(forward_kl.bias)))
        self.assertTrue(bool(torch.isfinite(reverse_kl.bias)))
        self.assertNotAlmostEqual(forward_kl.bias.item(), reverse_kl.bias.item())

    def test_padding_or_eos_cannot_be_a_training_target(self) -> None:
        model = PaperFaithfulDeDiff(
            num_nodes=3,
            dimension=4,
            rank=2,
            attention_heads=1,
            attention_head_dimension=4,
            dropout=0.0,
        )
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        output = model(
            torch.tensor([[0, 3]]),
            torch.zeros(1, 2),
            torch.tensor([1]),
            torch.zeros(1, 2, 2),
            edges,
            torch.ones(2),
            edges,
            torch.ones(2),
        )
        with self.assertRaisesRegex(ValueError, "real-user ids only"):
            paper_dediff_loss(output, torch.tensor([3]), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
