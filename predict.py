"""Track A entry point. Implement classify() before generating predictions."""

import csv
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "id", "customer", "bank", "account", "account_name", "account_type",
    "date", "amount_sek", "text",
}
VALID_CATEGORIES = {
    "income.employment", "income.public", "income.sales", "income.capital",
    "spend.housing", "spend.groceries", "spend.dining", "spend.transport",
    "spend.shopping", "spend.leisure", "spend.health", "spend.family",
    "spend.gambling", "credit.loan", "credit.collections", "transfer.own",
    "transfer.savings", "transfer.person", "transfer.cash", "unknown",
}


def classify(row: dict[str, str]) -> str:
    """Return one category from VALID_CATEGORIES for a single transaction."""
    raise NotImplementedError("Implement classify(row) for the case study")


def predict(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")

        predictions = []
        seen_ids = set()
        for row in reader:
            row_id = row["id"]
            if not row_id or row_id in seen_ids:
                raise ValueError(f"Missing or duplicate id: {row_id!r}")
            seen_ids.add(row_id)
            category = classify(row)
            if category not in VALID_CATEGORIES:
                raise ValueError(f"Invalid category for {row_id}: {category!r}")
            predictions.append({"id": row_id, "category": category})

    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "category"])
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py INPUT.csv")
    predict(Path(sys.argv[1]), Path("predictions.csv"))
