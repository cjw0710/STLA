from __future__ import annotations

import unittest

import numpy as np

from www2027.metrics import (
    active_user_churn,
    compute_drift_report,
    jensen_shannon_divergence,
    top_fraction_jaccard,
)


class DriftMetricsTest(unittest.TestCase):
    def test_identical_populations_have_no_distribution_drift(self) -> None:
        population = np.array([3.0, 2.0, 0.0, 1.0])
        self.assertAlmostEqual(jensen_shannon_divergence(population, population), 0.0)
        self.assertAlmostEqual(active_user_churn(population, population), 0.0)
        self.assertAlmostEqual(top_fraction_jaccard(population, population), 1.0)

    def test_disjoint_populations_have_maximum_base_two_jsd(self) -> None:
        first = np.array([1.0, 0.0, 0.0, 0.0])
        second = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(jensen_shannon_divergence(first, second), 1.0)
        self.assertAlmostEqual(active_user_churn(first, second), 1.0)
        self.assertAlmostEqual(top_fraction_jaccard(first, second), 0.0)

    def test_report_has_one_observation_per_transition(self) -> None:
        report = compute_drift_report(
            [np.array([1, 0]), np.array([1, 1]), np.array([0, 1])]
        )
        self.assertEqual(len(report.js_divergence), 2)
        self.assertEqual(len(report.top_hub_jaccard), 2)
        self.assertEqual(len(report.active_user_churn), 2)


if __name__ == "__main__":
    unittest.main()
