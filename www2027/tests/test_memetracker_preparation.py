from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from www2027.baselines.buzzbloom_temporal import TemporalBuzzLoader
from www2027.prepare_memetracker import parse_compact_timestamp


class MemeTrackerPreparationTest(unittest.TestCase):
    def test_compact_timestamp_restores_omitted_decimal_point(self) -> None:
        self.assertEqual(parse_compact_timestamp("3383039575"), 338303.9575)
        self.assertEqual(parse_compact_timestamp("338304029167"), 338304.029167)
        self.assertLess(
            parse_compact_timestamp("3383039575"),
            parse_compact_timestamp("338304029167"),
        )

    def test_strict_prepartitioned_loader_never_opens_test_json(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "sealed"
            dataset.mkdir()
            train = [
                {"cascade": [0, 1, 2], "timestamp": [float(i), float(i + 1), float(i + 2)]}
                for i in range(8)
            ]
            valid = [
                {"cascade": [1, 2, 3], "timestamp": [20.0, 21.0, 22.0]},
                {"cascade": [2, 3, 0], "timestamp": [23.0, 24.0, 25.0]},
            ]
            (dataset / "cascade_train.json").write_text(json.dumps(train), encoding="utf-8")
            (dataset / "cascade_valid.json").write_text(json.dumps(valid), encoding="utf-8")
            # Deliberately invalid JSON: constructing the selection loader must
            # still succeed, proving that the sealed partition is never opened.
            (dataset / "cascade_test.json").write_text("not valid json", encoding="utf-8")
            (dataset / "graph.txt").write_text("0,1\n1,2\n", encoding="utf-8")
            (dataset / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "split": {"strict_train_valid_only_during_selection": True},
                        "counts": {"all": 12, "train": 8, "valid": 2, "test": 2, "nodes": 4},
                    }
                ),
                encoding="utf-8",
            )

            loader = TemporalBuzzLoader(
                "sealed",
                root,
                max_prefix_length=5,
                valid_environments=2,
                mapping_root=root / "mappings",
            )
            self.assertTrue(loader.strict_prepartitioned)
            self.assertFalse(loader.test_materialized)
            self.assertFalse(hasattr(loader, "test_records"))
            self.assertEqual(loader.counts["test_retained_not_materialized"], 2)


if __name__ == "__main__":
    unittest.main()
