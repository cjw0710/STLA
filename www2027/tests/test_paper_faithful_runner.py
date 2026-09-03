from __future__ import annotations

import unittest

import numpy as np
import torch

from www2027.data import CascadeRecord
from www2027.train_paper_faithful import PaperPrefixDataset, remap_one_based_records


class PaperFaithfulRunnerTest(unittest.TestCase):
    def test_legacy_padding_offset_is_removed_from_real_users(self) -> None:
        raw = (
            CascadeRecord(
                cascade=(1, 3, 2),
                timestamp=(0.0, 1.0, 2.0),
                source_split="train",
                source_index=0,
            ),
        )
        remapped = remap_one_based_records(raw)
        self.assertEqual(remapped[0].cascade, (0, 2, 1))
        self.assertEqual(remapped[0].timestamp, raw[0].timestamp)

    def test_prefix_distance_contains_only_observed_prefix_users(self) -> None:
        record = CascadeRecord(
            cascade=(0, 2, 1),
            timestamp=(0.0, 1.0, 3.0),
            source_split="train",
            source_index=0,
        )
        distance = np.arange(9, dtype=np.float32).reshape(3, 3)
        dataset = PaperPrefixDataset(
            (record,),
            num_nodes=3,
            max_prefix_length=4,
            social_distance=distance,
        )
        # The second example predicts user 1 from prefix [0, 2].
        example = dataset[1]
        expected = torch.tensor([[0.0, 2.0], [6.0, 8.0]])
        self.assertTrue(torch.equal(example["social_distance"][:2, :2], expected))
        self.assertEqual(example["target"].item(), 1)
        self.assertEqual(example["prefix"][2].item(), 3)


if __name__ == "__main__":
    unittest.main()
