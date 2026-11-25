import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from claude_fallback import CATEGORIES, ClaudeFallback, MissingClaudeKey, load_api_key
from predict import REQUIRED_COLUMNS, VALID_CATEGORIES, predict


def row(text="Kortköp 260506 UBER *EATS HELP.UBER.COM"):
    return {"text": text, "account_name": "Everyday", "account_type": "checking", "amount_sek": "-100"}


class ClaudeFallbackTests(unittest.TestCase):
    def test_taxonomy_matches_submission_labels(self):
        self.assertEqual(set(CATEGORIES) | {"unknown"}, VALID_CATEGORIES)

    def test_key_is_loaded_from_environment_without_logging(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            self.assertEqual(load_api_key(), "test-key")

    def test_missing_key_fails_before_request(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with patch("claude_fallback.Path.exists", return_value=False):
                with self.assertRaises(MissingClaudeKey):
                    ClaudeFallback()

    def test_accepts_only_supported_high_certainty_labels(self):
        example = row()
        accepted = {"category": "spend.dining", "certainty": "high", "basis": "known_business", "evidence": "UBER"}
        self.assertEqual(ClaudeFallback._accepted(accepted, example), "spend.dining")
        for change in (
            {"certainty": "uncertain"},
            {"basis": "none"},
            {"evidence": "Kortköp"},
            {"evidence": "imaginary brand"},
            {"category": "invented"},
        ):
            self.assertEqual(ClaudeFallback._accepted(accepted | change, example), "unknown")

    def test_batch_deduplicates_and_preserves_row_order(self):
        llm = ClaudeFallback(api_key="test-key", batch_size=2, max_items=2)
        rows = [row(), row("Kortköp 260507 UBER *EATS HELP.UBER.COM"), row("LIDL GBG"), row("Unknown shop")]

        def fake_request(batch):
            return {"results": [
                {"index": index, "category": "spend.dining" if "UBER" in item["text"] else "spend.groceries",
                 "certainty": "high", "basis": "known_business", "evidence": "UBER" if "UBER" in item["text"] else "LIDL"}
                for index, item in enumerate(batch)
            ]}

        with patch.object(llm, "_request", side_effect=fake_request) as request:
            self.assertEqual(llm.classify_many(rows), ["spend.dining", "spend.dining", "spend.groceries", "unknown"])
            self.assertEqual(request.call_count, 1)

    def test_pipeline_uses_claude_only_for_remaining_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "output.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
                writer.writeheader()
                for row_id, text, amount in (("a", "LÖN", "30000"), ("b", "unknown shop", "-100")):
                    writer.writerow({
                        "id": row_id, "customer": "c", "bank": "bank", "account": "main",
                        "account_name": "Everyday", "account_type": "checking", "date": "2026-05-25",
                        "amount_sek": amount, "text": text,
                    })
            with patch("predict.MerchantMatcher") as matcher, patch("predict.TextClassifier") as text_model, patch("predict.ClaudeFallback") as claude:
                matcher.return_value.match.return_value.category = "unknown"
                text_model.return_value.classify.return_value = "unknown"
                claude.return_value.classify_many.return_value = ["spend.shopping"]
                claude.return_value.calls = 1
                claude.return_value.input_tokens = 100
                claude.return_value.output_tokens = 20
                predict(source, output, text_model=True, claude=True)
                self.assertEqual(matcher.return_value.match.call_count, 1)
                self.assertEqual(text_model.return_value.classify.call_count, 1)
                self.assertEqual(len(claude.return_value.classify_many.call_args.args[0]), 1)
            with output.open(encoding="utf-8") as stream:
                self.assertEqual([r["category"] for r in csv.DictReader(stream)], ["income.employment", "spend.shopping"])


if __name__ == "__main__":
    unittest.main()
