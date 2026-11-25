"""Train-only merchant lookup for transaction descriptions."""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher


BANK_PREFIX = re.compile(r"^(?:KORTKOP\s+\d{6}\s+|AUTOGIRO\s+|BETALNING\s+BG\s+|BG\s+)")
SWISH_PREFIX = re.compile(r"^SWISH(?:\s+(?:BETALNING|INBETALNING|FRAN|TILL))?\s+")
PROVIDER_PREFIX = re.compile(r"^(?:KLARNA|ZETTLE_?|SUMUP|IZ|PAYPAL|TRUSTLY)\s*\*\s*")
LONG_REFERENCE = re.compile(r"\s+(?=[A-Z0-9]{8,}$)(?=[A-Z0-9]*\d)[A-Z0-9]+$")


def normalize_merchant(text: str) -> str:
    """Remove bank formatting while retaining the business and product words."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper().strip()
    value = BANK_PREFIX.sub("", value)
    value = SWISH_PREFIX.sub("", value)
    value = PROVIDER_PREFIX.sub("", value)
    value = re.sub(r"\bHELP\.UBER\.COM\b", " ", value)
    value = re.sub(r"\b(?:EUR|USD|GBP|DKK)\s+\d+(?:[.,]\d+)?\b", " ", value)
    value = re.sub(r"\b(?:UTBETALNING|ATERBETALNING|REFUND|REVERSAL)\b", " ", value)
    value = LONG_REFERENCE.sub(" ", value)
    value = re.sub(r"\b[A-Z]{1,3}\d{3,}\b", " ", value)
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return " ".join(value.split())


@dataclass(frozen=True)
class MatchResult:
    category: str = "unknown"
    source: str = "unknown"
    similarity: float = 0.0
    merchant: str = ""


class MerchantMatcher:
    """Match spending merchants learned from labelled training rows only."""

    def __init__(
        self,
        training_rows: list[dict[str, str]],
        *,
        mode: str = "full",
        similarity_threshold: float = 0.88,
        category_margin: float = 0.10,
    ) -> None:
        if mode not in {"none", "exact", "full"}:
            raise ValueError(f"Invalid matching mode: {mode}")
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.category_margin = category_margin
        labels = defaultdict(Counter)
        for row in training_rows:
            if row["category"].startswith("spend."):
                merchant = normalize_merchant(row["text"])
                if merchant:
                    labels[merchant][row["category"]] += 1

        self.exact = {
            merchant: next(iter(counts)) if len(counts) == 1 else None
            for merchant, counts in labels.items()
        }
        self.candidates = tuple(sorted(
            (merchant, category)
            for merchant, category in self.exact.items()
            if category is not None
        ))

        by_prefix = defaultdict(lambda: defaultdict(set))
        for merchant, category in self.candidates:
            first = merchant.split()[0]
            if len(first) >= 3:
                by_prefix[first][category].add(merchant)
        self.chains = {
            prefix: next(iter(categories))
            for prefix, categories in by_prefix.items()
            if len(categories) == 1 and len(next(iter(categories.values()))) >= 2
        }
        self.cache: dict[str, MatchResult] = {}

    def match(self, row: dict[str, str]) -> MatchResult:
        if self.mode == "none" or row["text"].strip().casefold().startswith(("överföring", "overforing")):
            return MatchResult()

        merchant = normalize_merchant(row["text"])
        if not merchant:
            return MatchResult()
        if merchant in self.exact:
            category = self.exact[merchant]
            return MatchResult(category, "exact", 1.0, merchant) if category else MatchResult(merchant=merchant)

        # An incoming payment may be a refund, but a loose name match could be income.
        # Swish to a person should not be fuzzily matched to a business.
        if self.mode == "exact" or Decimal(row["amount_sek"]) > 0 or row["text"].strip().upper().startswith("SWISH"):
            return MatchResult(merchant=merchant)

        if merchant in self.cache:
            return self.cache[merchant]
        extensions = {
            category
            for candidate, category in self.candidates
            if len(candidate.split()) >= 2 and merchant.startswith(candidate + " ")
        }
        prefix = merchant.split()[0]
        if len(extensions) == 1:
            result = MatchResult(next(iter(extensions)), "extension", merchant=merchant)
        elif extensions:
            result = MatchResult(merchant=merchant)
        elif prefix in self.chains:
            result = MatchResult(self.chains[prefix], "chain", merchant=merchant)
        elif len(merchant.replace(" ", "")) >= 5:
            scores = sorted(
                (
                    (SequenceMatcher(None, merchant, candidate).ratio(), category)
                    for candidate, category in self.candidates
                ),
                reverse=True,
            )
            if scores:
                best_score, best_category = scores[0]
                other_score = max((score for score, category in scores if category != best_category), default=0.0)
                if best_score >= self.similarity_threshold and best_score - other_score >= self.category_margin:
                    result = MatchResult(best_category, "fuzzy", best_score, merchant)
                else:
                    result = MatchResult(merchant=merchant)
            else:
                result = MatchResult(merchant=merchant)
        else:
            result = MatchResult(merchant=merchant)
        self.cache[merchant] = result
        return result
