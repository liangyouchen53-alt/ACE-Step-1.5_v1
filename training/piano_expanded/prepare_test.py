"""Checks for invalid audio exclusion and train/holdout separation."""

import unittest

import numpy as np

from prepare import choose_windows, validate_splits


class PrepareTest(unittest.TestCase):
    """Exercise safe window selection and split guards."""

    def test_windows(self):
        audio = np.zeros((120, 1), dtype=np.float32)
        audio[20:40] = 0.05
        audio[40:60] = 1.0
        audio[60:80] = np.nan
        audio[80:100] = 0.1
        self.assertEqual([start for _, start in choose_windows(audio, 1, 8)], [20, 80])

    def test_leakage(self):
        records = [{"split": "train", "sha256": "a", "work": "a"},
                   {"split": "heldout", "sha256": "b", "work": "b"}]
        validate_splits(records)
        records[1]["work"] = "a"
        with self.assertRaises(ValueError):
            validate_splits(records)


if __name__ == "__main__":
    unittest.main()
