"""Score a predictions CSV against a labelled local dataset."""

import argparse
import csv
from pathlib import Path

from predict import VALID_CATEGORIES


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
        return list(reader)


def score(gold_path: Path, predictions_path: Path) -> None:
    gold_rows = read_rows(gold_path, {"id", "category"})
    prediction_rows = read_rows(predictions_path, {"id", "category"})
    gold = {row["id"]: row["category"] for row in gold_rows}
    predictions = {row["id"]: row["category"] for row in prediction_rows}
    if len(gold) != len(gold_rows) or len(predictions) != len(prediction_rows):
        raise ValueError("Duplicate IDs in gold or predictions")
    if not gold:
        raise ValueError("Gold dataset is empty")
    extra = predictions.keys() - gold.keys()
    if extra:
        raise ValueError(f"Predictions contain unknown IDs: {', '.join(sorted(extra)[:5])}")
    invalid = set(predictions.values()) - VALID_CATEGORIES
    if invalid:
        raise ValueError(f"Invalid categories: {', '.join(sorted(invalid))}")

    right = wrong = 0
    for row_id, true_category in gold.items():
        predicted = predictions.get(row_id, "unknown")
        if predicted == "unknown":
            continue
        if predicted == true_category:
            right += 1
        else:
            wrong += 1
    total = len(gold)
    labelled = right + wrong
    print(f"rows={total} right={right} wrong={wrong} unknown={total - labelled}")
    print(f"score={100 * (right - wrong) / total:.2f} per 100 rows")
    print(f"accuracy={right / total:.2%}")
    print(f"precision={right / labelled:.2%}" if labelled else "precision=n/a")
    print(f"coverage={labelled / total:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path, help="labelled CSV, such as data/transactions.csv")
    parser.add_argument("predictions", type=Path, help="CSV with id,category columns")
    args = parser.parse_args()
    score(args.gold, args.predictions)
