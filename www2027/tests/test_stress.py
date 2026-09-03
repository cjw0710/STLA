from __future__ import annotations

import unittest

import torch

from www2027.stress import perturb_recent_popularity


class PopularityStressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.historical = torch.tensor([10.0, 8.0, 4.0, 2.0, 0.0, 0.0])
        self.recent = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.0, 0.0])

    def test_zero_severity_is_identity(self) -> None:
        for stress in (
            "recent_hub_amplification",
            "recent_hub_turnover",
            "emerging_influx",
        ):
            shifted = perturb_recent_popularity(
                self.historical, self.recent, stress, 0.0
            )
            self.assertTrue(torch.equal(shifted, self.recent))

    def test_hub_turnover_conserves_recent_mass(self) -> None:
        shifted = perturb_recent_popularity(
            self.historical,
            self.recent,
            "recent_hub_turnover",
            1.0,
        )
        self.assertAlmostEqual(shifted.sum().item(), self.recent.sum().item())
        self.assertLess(shifted[0].item(), self.recent[0].item())
        self.assertGreater(shifted[1:].sum().item(), self.recent[1:].sum().item())

    def test_emerging_influx_activates_unseen_users(self) -> None:
        shifted = perturb_recent_popularity(
            self.historical,
            self.recent,
            "emerging_influx",
            0.5,
        )
        self.assertAlmostEqual(shifted.sum().item(), self.recent.sum().item())
        self.assertGreater(shifted[4:].sum().item(), 0.0)

    def test_hub_amplification_increases_total_recent_mass(self) -> None:
        shifted = perturb_recent_popularity(
            self.historical,
            self.recent,
            "recent_hub_amplification",
            1.0,
        )
        self.assertGreater(shifted.sum().item(), self.recent.sum().item())

    def test_invalid_severity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "severity"):
            perturb_recent_popularity(
                self.historical,
                self.recent,
                "emerging_influx",
                1.1,
            )


if __name__ == "__main__":
    unittest.main()
