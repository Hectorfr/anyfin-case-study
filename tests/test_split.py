import csv
import unittest
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from make_split import description_key, split


DATA = Path(__file__).resolve().parents[1] / "data"


def read(name):
    with (DATA / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read("transactions.csv")
        cls.train = read("train.csv")
        cls.dev = read("dev.csv")

    def test_fixed_files_cover_source_exactly(self):
        generated_train, generated_dev = split(self.source)
        self.assertEqual(self.train, generated_train)
        self.assertEqual(self.dev, generated_dev)
        self.assertEqual(len(self.train) + len(self.dev), len(self.source))
        self.assertFalse({r["id"] for r in self.train} & {r["id"] for r in self.dev})

    def test_repeated_descriptions_stay_together(self):
        train_keys = {description_key(r["text"]) for r in self.train}
        dev_keys = {description_key(r["text"]) for r in self.dev}
        self.assertFalse(train_keys & dev_keys)

    def test_transfer_pairs_stay_together(self):
        side = {r["id"]: "train" for r in self.train}
        side.update({r["id"]: "dev" for r in self.dev})
        groups = defaultdict(list)
        for row in self.source:
            if row["text"].strip().casefold().startswith(("överföring", "overforing")):
                groups[(row["customer"], row["date"], abs(Decimal(row["amount_sek"])))].append(row["id"])
        for ids in groups.values():
            self.assertEqual(len({side[row_id] for row_id in ids}), 1)


if __name__ == "__main__":
    unittest.main()
