import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from merchant_matching import MatchResult, MerchantMatcher, normalize_merchant
from predict import REQUIRED_COLUMNS, classify, predict


def training_row(text, category):
    return {"text": text, "category": category}


def query(text, amount="-100"):
    return {"text": text, "amount_sek": amount}


class MerchantNormalizationTests(unittest.TestCase):
    def test_bank_and_provider_wrappers_leave_the_business(self):
        self.assertEqual(normalize_merchant("Kortköp 260917 KLARNA*ARKET"), "ARKET")
        self.assertEqual(normalize_merchant("Kortköp 260521 ZETTLE_*KAFE SOLROS"), "KAFE SOLROS")
        self.assertEqual(normalize_merchant("PAYPAL *STEAMGAMES"), "STEAMGAMES")
        self.assertEqual(normalize_merchant("IZ *KAFE LEVE"), "KAFE LEVE")

    def test_distinguishes_uber_products(self):
        self.assertEqual(normalize_merchant("Kortköp 260618 UBER *TRIP HELP.UBER.COM"), "UBER TRIP")
        self.assertEqual(normalize_merchant("Kortköp 260506 UBER *EATS HELP.UBER.COM"), "UBER EATS")

    def test_payout_and_currency_amount_are_not_merchants(self):
        self.assertEqual(normalize_merchant("TRUSTLY*UNIBET UTBETALNING"), "UNIBET")
        self.assertEqual(normalize_merchant("NETTO VEJLE DKK 416.12"), "NETTO VEJLE")
        self.assertEqual(normalize_merchant("SPOTIFY P2F9A1C3"), "SPOTIFY")
        self.assertEqual(normalize_merchant("REMA1000 VEJLE"), "REMA1000 VEJLE")


class MerchantMatcherTests(unittest.TestCase):
    def test_exact_match_and_refund(self):
        matcher = MerchantMatcher([training_row("Kortköp 260501 ZALANDO SE", "spend.shopping")])
        self.assertEqual(matcher.match(query("ZALANDO SE")).category, "spend.shopping")
        self.assertEqual(matcher.match(query("ZALANDO SE", "100")).category, "spend.shopping")

    def test_chain_requires_multiple_consistent_store_names(self):
        matcher = MerchantMatcher([
            training_row("WILLYS SOLNA", "spend.groceries"),
            training_row("WILLYS GBG HISINGEN", "spend.groceries"),
        ])
        result = matcher.match(query("WILLYS MALMO"))
        self.assertEqual((result.category, result.source), ("spend.groceries", "chain"))

    def test_conflicting_labels_and_ambiguous_prefix_abstain(self):
        matcher = MerchantMatcher([
            training_row("SAME SHOP", "spend.shopping"),
            training_row("SAME SHOP", "spend.dining"),
            training_row("UBER TRIP", "spend.transport"),
            training_row("UBER EATS", "spend.dining"),
        ])
        self.assertEqual(matcher.match(query("SAME SHOP")).category, "unknown")
        self.assertEqual(matcher.match(query("UBER UNKNOWN SERVICE")).category, "unknown")

    def test_one_character_spelling_variant(self):
        matcher = MerchantMatcher([
            training_row("ZALANDO SE", "spend.shopping"),
            training_row("WILLYS SOLNA", "spend.groceries"),
        ])
        result = matcher.match(query("ZALNDO SE"))
        self.assertEqual((result.category, result.source), ("spend.shopping", "fuzzy"))

    def test_known_multiword_merchant_with_location_extension(self):
        matcher = MerchantMatcher([
            training_row("UBER *TRIP HELP.UBER.COM", "spend.transport"),
            training_row("UBER *EATS HELP.UBER.COM", "spend.dining"),
        ])
        self.assertEqual(matcher.match(query("UBER *TRIP LISBOA")).category, "spend.transport")
        self.assertEqual(matcher.match(query("UBER *EATS LISBOA")).category, "spend.dining")

    def test_no_loose_match_on_incoming_or_person_transfer(self):
        matcher = MerchantMatcher([
            training_row("ZALANDO SE", "spend.shopping"),
            training_row("SALLADSBAREN AB", "spend.dining"),
        ])
        self.assertEqual(matcher.match(query("ZALNDO SE", "100")).category, "unknown")
        self.assertEqual(matcher.match(query("Swish betalning SALLADSBAREN AC")).category, "unknown")
        self.assertEqual(matcher.match(query("Överföring ZALANDO SE")).category, "unknown")

    def test_rules_keep_priority_over_merchant_names(self):
        matcher = MerchantMatcher([
            training_row("KLARNA*BOOZT", "spend.shopping"),
        ])
        loan = {
            "text": "Betalning BG KLARNA AB",
            "amount_sek": "-500",
            "account_type": "checking",
            "account_name": "Everyday",
        }
        self.assertEqual(classify(loan), "credit.loan")
        self.assertEqual(matcher.match(query("KLARNA*BOOZT")).category, "spend.shopping")

    def test_prediction_loop_calls_matcher_only_after_unknown_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "predictions.csv"
            fields = sorted(REQUIRED_COLUMNS)
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row_id, text in (("a", "LÖN"), ("b", "ZALANDO SE")):
                    writer.writerow({
                        "id": row_id, "customer": "c-test", "bank": "Example",
                        "account": "main", "account_name": "Everyday", "account_type": "checking",
                        "date": "2026-05-25", "amount_sek": "100" if row_id == "a" else "-100",
                        "text": text,
                    })
            with patch("predict.MerchantMatcher") as matcher_type:
                matcher_type.return_value.match.return_value = MatchResult("spend.shopping", "exact", 1.0, "ZALANDO SE")
                predict(source, output)
                matcher_type.return_value.match.assert_called_once()
            with output.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [
                {"id": "a", "category": "income.employment"},
                {"id": "b", "category": "spend.shopping"},
            ])


if __name__ == "__main__":
    unittest.main()
