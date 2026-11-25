import unittest

from predict import classify, paired_transfer_categories


def row(row_id, amount, text, account="main", name="Everyday", account_type="checking"):
    return {
        "id": row_id,
        "customer": "c-test",
        "bank": "Example Bank",
        "account": account,
        "account_name": name,
        "account_type": account_type,
        "date": "2026-05-25",
        "amount_sek": str(amount),
        "text": text,
    }


class RuleTests(unittest.TestCase):
    def test_salary_requires_incoming_money(self):
        self.assertEqual(classify(row("a", "30000", "LÖN")), "income.employment")
        self.assertEqual(classify(row("b", "-30000", "LÖN")), "unknown")

    def test_matching_checking_transfer_is_own(self):
        rows = [row("a", "-500", "Överföring Household"), row("b", "500", "Överföring Everyday", "house", "Household")]
        paired = paired_transfer_categories(rows)
        self.assertEqual([classify(r, paired.get(r["id"])) for r in rows], ["transfer.own"] * 2)

    def test_buffer_account_is_savings_even_when_typed_checking(self):
        rows = [row("a", "-500", "Överföring 123"), row("b", "500", "Överföring 456", "buffer", "Buffert")]
        paired = paired_transfer_categories(rows)
        self.assertEqual([classify(r, paired.get(r["id"])) for r in rows], ["transfer.savings"] * 2)

    def test_credit_card_payment_has_different_labels_on_each_side(self):
        rows = [row("a", "-500", "Överföring Kreditkort"), row("b", "500", "Överföring Everyday", "card", "Credit card", "credit_card")]
        paired = paired_transfer_categories(rows)
        self.assertEqual([classify(r, paired.get(r["id"])) for r in rows], ["credit.loan", "transfer.own"])

    def test_payment_provider_purchase_is_not_loan_invoice(self):
        self.assertEqual(classify(row("a", "-500", "KLARNA*ELLOS")), "unknown")
        self.assertEqual(classify(row("b", "-500", "Betalning BG KLARNA AB")), "credit.loan")

    def test_collections_and_cash_are_distinct_from_loans(self):
        self.assertEqual(classify(row("a", "-500", "BG KRONOFOGDEN")), "credit.collections")
        self.assertEqual(classify(row("b", "-500", "ATM BANKOMAT GBG")), "transfer.cash")

    def test_unmatched_everyday_transfer_abstains(self):
        self.assertEqual(classify(row("a", "-500", "Överföring okänt konto")), "unknown")


if __name__ == "__main__":
    unittest.main()
