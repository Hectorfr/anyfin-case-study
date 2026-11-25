import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from predict import REQUIRED_COLUMNS, predict
from text_classifier import TextClassifier


class TextClassifierTests(unittest.TestCase):
    def test_features_include_merchant_account_and_direction(self):
        row = {"text": "Kortköp 260521 ZETTLE_*KAFE SOLROS", "account_type": "credit_card", "amount_sek": "-75.00"}
        self.assertEqual(TextClassifier.features(row), "KAFE SOLROS ACCOUNT_CREDIT_CARD OUTGOING")

    def test_fallback_runs_only_after_rule_and_merchant_are_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "predictions.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
                writer.writeheader()
                for row_id, text, amount in (("a", "LÖN", "30000"), ("b", "unknown business", "-100")):
                    writer.writerow({
                        "id": row_id, "customer": "c-test", "bank": "Example", "account": "main",
                        "account_name": "Everyday", "account_type": "checking", "date": "2026-05-25",
                        "amount_sek": amount, "text": text,
                    })
            with patch("predict.MerchantMatcher") as matcher_type, patch("predict.TextClassifier") as model_type:
                matcher_type.return_value.match.return_value.category = "unknown"
                model_type.return_value.classify.return_value = "spend.shopping"
                predict(source, output, text_model=True)
                self.assertEqual(matcher_type.return_value.match.call_count, 1)
                self.assertEqual(model_type.return_value.classify.call_count, 1)
            with output.open(encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [
                    {"id": "a", "category": "income.employment"},
                    {"id": "b", "category": "spend.shopping"},
                ])


if __name__ == "__main__":
    unittest.main()
