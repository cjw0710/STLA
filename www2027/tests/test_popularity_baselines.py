import unittest

import torch

from www2027.evaluate_popularity_baselines import prefix_masked_scores


class PopularityBaselineTest(unittest.TestCase):
    def test_prefix_masking_is_causal_and_keeps_future_users(self) -> None:
        base = torch.tensor([4.0, 3.0, 2.0, 1.0])
        # Real users are shifted by two; zero is padding.
        sequence = torch.tensor([[2, 4, 5, 0]])
        scores = prefix_masked_scores(base, sequence).reshape(1, 3, 4)
        self.assertTrue(torch.isneginf(scores[0, 0, 0]))
        self.assertEqual(float(scores[0, 0, 2]), 2.0)
        self.assertTrue(torch.isneginf(scores[0, 1, 0]))
        self.assertTrue(torch.isneginf(scores[0, 1, 2]))
        self.assertEqual(float(scores[0, 1, 3]), 1.0)
        self.assertTrue(torch.isneginf(scores[0, 2, 3]))


if __name__ == "__main__":
    unittest.main()
