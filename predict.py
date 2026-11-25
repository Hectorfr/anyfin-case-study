"""Conservative rule-based baseline for the Anyfin Track A case."""

import argparse
import csv
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from merchant_matching import MerchantMatcher


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


def is_transfer(row: dict[str, str]) -> bool:
    return row["text"].strip().casefold().startswith(("överföring", "overforing"))


def is_savings_account(row: dict[str, str]) -> bool:
    return row["account_type"].casefold() == "savings" or bool(
        re.search(r"\b(?:spar|sparkonto|buffert)\b", row["account_name"].casefold())
    )


def paired_transfer_categories(rows: list[dict[str, str]]) -> dict[str, str]:
    """Label unambiguous, opposite entries within a customer's accounts."""
    groups = defaultdict(list)
    for row in rows:
        amount = Decimal(row["amount_sek"])
        if amount and is_transfer(row):
            groups[(row["customer"], row["date"], abs(amount))].append(row)

    paired = {}
    for group in groups.values():
        if len(group) != 2:
            continue
        first, second = group
        if first["account"] == second["account"]:
            continue
        if Decimal(first["amount_sek"]) != -Decimal(second["amount_sek"]):
            continue
        if (first["account_type"] == "credit_card") != (second["account_type"] == "credit_card"):
            card = first if first["account_type"] == "credit_card" else second
            payment = second if card is first else first
            if Decimal(card["amount_sek"]) > 0 and Decimal(payment["amount_sek"]) < 0:
                paired[card["id"]] = "transfer.own"
                paired[payment["id"]] = "credit.loan"
            continue
        category = "transfer.savings" if is_savings_account(first) or is_savings_account(second) else "transfer.own"
        paired[first["id"]] = category
        paired[second["id"]] = category
    return paired


def classify(row: dict[str, str], paired_category: str | None = None) -> str:
    """Return a category only when a narrow rule applies; otherwise abstain."""
    text = row["text"].strip().upper()
    amount = Decimal(row["amount_sek"])

    if amount > 0 and re.search(r"\b(?:LÖN|LON|SALARY)\b", text):
        return "income.employment"

    if paired_category:
        return paired_category
    if is_transfer(row) and (
        is_savings_account(row)
        or re.search(r"\b(?:SPAR|SPARKONTO|BUFFERT)\b", text)
    ):
        return "transfer.savings"
    if amount > 0 and row["account_type"].casefold() == "credit_card" and text == "INBETALNING TACK":
        return "transfer.own"

    if amount < 0:
        if re.search(r"\b(?:INTRUM|KRONOFOGDEN)\b", text):
            return "credit.collections"
        if re.search(r"\b(?:UTTAG|ATM)\s+BANKOMAT\b", text):
            return "transfer.cash"
        if re.search(r"\b(?:BOLÅN|HYPOTEK|AMORTERING|LÅNEBETALNING)\b", text):
            return "credit.loan"
        if re.search(r"\b(?:KLARNA|QLIRO|WALLEY)\s+AB\b", text):
            return "credit.loan"
        if re.search(r"\b(?:CSN|SANTANDER CONSUMER|RESURS BANK|NORDAX BANK|ANYFIN AB|SEB KORT MASTERCARD)\b", text) and re.match(
            r"^(?:AUTOGIRO|BG|BETALNING BG)\b", text
        ):
            return "credit.loan"

    return "unknown"


def predict(input_path: Path, output_path: Path, *, matching: str = "full") -> None:
    training_path = Path(__file__).resolve().parent / "data/train.csv"
    with training_path.open(newline="", encoding="utf-8-sig") as training_source:
        matcher = MerchantMatcher(list(csv.DictReader(training_source)), mode=matching)

    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")

        rows = list(reader)
        paired = paired_transfer_categories(rows)
        predictions = []
        seen_ids = set()
        for row in rows:
            row_id = row["id"]
            if not row_id or row_id in seen_ids:
                raise ValueError(f"Missing or duplicate id: {row_id!r}")
            seen_ids.add(row_id)
            category = classify(row, paired.get(row_id))
            if category == "unknown":
                category = matcher.match(row).category
            if category not in VALID_CATEGORIES:
                raise ValueError(f"Invalid category for {row_id}: {category!r}")
            predictions.append({"id": row_id, "category": category})

    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "category"])
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV to label")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--matching", choices=("none", "exact", "full"), default="full")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predict(args.input, args.output, matching=args.matching)
