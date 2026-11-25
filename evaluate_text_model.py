"""Choose the optional text model threshold using train-only description groups."""

import csv
from pathlib import Path

from make_split import description_key
from merchant_matching import MerchantMatcher
from predict import classify, paired_transfer_categories
from text_classifier import TextClassifier


def main() -> None:
    from sklearn.model_selection import GroupKFold

    with (Path(__file__).resolve().parent / "data/train.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    groups = [description_key(row["text"]) for row in rows]
    observations = []

    for train_indices, held_indices in GroupKFold(n_splits=3).split(rows, groups=groups):
        training = [rows[index] for index in train_indices]
        held = [rows[index] for index in held_indices]
        matcher = MerchantMatcher(training)
        model = TextClassifier(training)
        paired = paired_transfer_categories(held)
        for row in held:
            category = classify(row, paired.get(row["id"]))
            if category == "unknown":
                category = matcher.match(row).category
            if category == "unknown":
                label, score = model.top_prediction(row)
                observations.append((row["category"], label, score))

    results = []
    for integer in range(10, 100, 5):
        threshold = integer / 100
        labelled = [(truth, label) for truth, label, score in observations if score >= threshold]
        right = sum(truth == label for truth, label in labelled)
        wrong = len(labelled) - right
        results.append((right - wrong, threshold, right, wrong))
    net, threshold, right, wrong = max(results)
    print(f"Train-only grouped folds: {len(observations)} fallback rows")
    print(f"Best threshold={threshold:.2f}, right={right}, wrong={wrong}, net={net}")
    print("Ties prefer the higher threshold; the model score is not calibrated.")


if __name__ == "__main__":
    main()
