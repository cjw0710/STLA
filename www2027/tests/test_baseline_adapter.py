from __future__ import annotations

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from www2027.baselines.buzzbloom_temporal import (
    EOS,
    PAD,
    PAD_TIME,
    TemporalBuzzLoader,
    load_model_class,
    make_collate_fn,
    vectorized_mshgat_previous_user_mask,
)
from www2027.data import CascadeRecord


class BuzzBloomTemporalAdapterTest(unittest.TestCase):
    def test_collate_caps_sequence_and_uses_distinct_padding(self) -> None:
        collate = make_collate_fn(max_prefix_length=2)
        sequence, timestamp, index = collate(
            [
                ([2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0], 7),
                ([6, 7], [5.0, 6.0], 8),
            ]
        )
        self.assertEqual(tuple(sequence.shape), (2, 3))
        self.assertEqual(sequence[0].tolist(), [2, 3, 4])
        self.assertEqual(sequence[1].tolist(), [6, 7, PAD])
        self.assertEqual(timestamp[1].tolist(), [5.0, 6.0, PAD_TIME])
        self.assertEqual(index.tolist(), [7, 8])

    def test_record_shift_does_not_conflate_real_user_zero_with_pad(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "toy"
            dataset_dir.mkdir()
            rows = [
                {
                    "cascade": [index % 4, (index + 1) % 4, (index + 2) % 4],
                    "timestamp": [float(index), float(index) + 1, float(index) + 2],
                }
                for index in range(30)
            ]
            for split_name, subset in zip(
                ("train", "valid", "test"),
                (rows[:10], rows[10:20], rows[20:]),
            ):
                import json

                (dataset_dir / f"cascade_{split_name}.json").write_text(
                    json.dumps(subset),
                    encoding="utf-8",
                )
            (dataset_dir / "graph.txt").write_text("0,1\n1,2\n", encoding="utf-8")

            mapping_root = root / "mappings"
            loader = TemporalBuzzLoader(
                "toy",
                root,
                max_prefix_length=5,
                valid_environments=2,
                mapping_root=mapping_root,
            )

            self.assertFalse(loader.test_materialized)
            self.assertFalse(hasattr(loader, "test_set"))
            self.assertEqual(loader.user_num, 6)
            self.assertEqual(loader.counts, {
                "all": 30,
                "train": 21,
                "valid": 3,
                "test_retained_not_materialized": 6,
            })
            self.assertGreaterEqual(min(loader.train_set.cascades[0]), 2)
            self.assertNotIn(EOS, loader.train_set.cascades[0])
            self.assertTrue(all(index == 0 for dataset in loader.valid_sets for index in dataset.indices))
            self.assertEqual(len(loader.cascades), len(loader.train_set))
            self.assertTrue(all(cascade[-1] == EOS for cascade in loader.cascades))

            with Path(loader.u2idx_dict).open("rb") as stream:
                u2idx = pickle.load(stream)
            self.assertEqual(u2idx["<blank>"], PAD)
            self.assertEqual(u2idx["</s>"], EOS)
            self.assertEqual(u2idx["0"], 2)

    def test_malformed_short_cascade_is_rejected_at_collation(self) -> None:
        collate = make_collate_fn(max_prefix_length=5)
        with self.assertRaises(ValueError):
            collate([([2], [1.0], 1)])

    def test_vectorized_mshgat_mask_is_exactly_equal_to_upstream(self) -> None:
        Model = load_model_class("MSHGAT")
        sequence = torch.tensor(
            [[2, 3, 2, 0], [4, 5, 6, 7]],
            dtype=torch.long,
        )
        expected = Model.get_previous_user_mask(
            None,
            sequence,
            user_size=9,
            device=torch.device("cpu"),
        )
        actual = vectorized_mshgat_previous_user_mask(
            None,
            sequence,
            user_size=9,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_cascade_record_fixture_still_rejects_time_reversal(self) -> None:
        with self.assertRaises(ValueError):
            CascadeRecord((0, 1), (2.0, 1.0), "toy", 0)


if __name__ == "__main__":
    unittest.main()
