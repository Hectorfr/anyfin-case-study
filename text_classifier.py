"""Optional local text fallback trained only on the labelled training split."""

from decimal import Decimal

from merchant_matching import normalize_merchant


class MissingTextModelDependency(RuntimeError):
    """The optional local model's dependency is not installed."""


class TextClassifier:
    """Character n-gram TF-IDF and logistic regression with conservative abstention."""

    def __init__(self, training_rows: list[dict[str, str]], *, threshold: float = 0.60) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:
            raise MissingTextModelDependency(
                "The optional text model needs scikit-learn. Install requirements-model.txt first."
            ) from exc

        self.threshold = threshold
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=2.0, random_state=0),
        )
        self.model.fit(
            [self.features(row) for row in training_rows],
            [row["category"] for row in training_rows],
        )
        self.classes = self.model[-1].classes_
        self.cache: dict[str, tuple[str, float]] = {}

    @staticmethod
    def features(row: dict[str, str]) -> str:
        direction = "INCOMING" if Decimal(row["amount_sek"]) > 0 else "OUTGOING"
        return f"{normalize_merchant(row['text'])} ACCOUNT_{row['account_type'].upper()} {direction}"

    def top_prediction(self, row: dict[str, str]) -> tuple[str, float]:
        """Return the top label and its raw model score (not a calibrated probability)."""
        features = self.features(row)
        if features not in self.cache:
            scores = self.model.predict_proba([features])[0]
            best = int(scores.argmax())
            self.cache[features] = str(self.classes[best]), float(scores[best])
        return self.cache[features]

    def classify(self, row: dict[str, str]) -> str:
        label, score = self.top_prediction(row)
        return label if score >= self.threshold else "unknown"
