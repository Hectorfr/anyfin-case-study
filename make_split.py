"""Create a fixed, description-grouped development split from the labelled ledger."""

import csv
import hashlib
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


SOURCE = Path("data/transactions.csv")
TRAIN = Path("data/train.csv")
DEV = Path("data/dev.csv")
DEV_FRACTION = 0.20
SEED = "anyfin-dev-v1"


def description_key(text: str) -> str:
    """Group repeated bank descriptions while discarding dates and account digits."""
    text = text.casefold()
    text = re.sub(r"\b(?:kortköp|kortkop|autogiro|betalning|bg)\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\wåäö]+", " ", text)
    return " ".join(text.split())


def split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        parent[find(b)] = find(a)

    by_description = {}
    transfer_pairs = defaultdict(list)
    for index, row in enumerate(rows):
        key = description_key(row["text"])
        if key in by_description:
            union(index, by_description[key])
        else:
            by_description[key] = index

        if row["text"].strip().casefold().startswith(("överföring", "overforing")):
            amount = Decimal(row["amount_sek"])
            transfer_pairs[(row["customer"], row["date"], abs(amount))].append(index)

    # A transfer's two ledger entries must land in the same split.
    for indices in transfer_pairs.values():
        for index in indices[1:]:
            union(indices[0], index)

    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)

    def order(indices: list[int]) -> bytes:
        stable_id = min(rows[index]["id"] for index in indices)
        return hashlib.sha256(f"{SEED}:{stable_id}".encode()).digest()

    component_list = list(components.values())
    component_counts = [
        Counter(rows[index]["category"] for index in indices) for indices in component_list
    ]
    component_categories = [
        set(counts) for counts in component_counts
    ]
    by_category = defaultdict(list)
    for component_index, categories in enumerate(component_categories):
        for category in categories:
            by_category[category].append(component_index)
    remaining = {category: len(indices) for category, indices in by_category.items()}

    target = round(len(rows) * DEV_FRACTION)
    dev_indices = set()
    selected_components = set()
    dev_counts = Counter()
    source_counts = Counter(row["category"] for row in rows)
    category_targets = {
        category: count * DEV_FRACTION
        for category, count in source_counts.items()
        if len(by_category[category]) > 1
    }

    def can_select(component_index: int) -> bool:
        return (
            component_index not in selected_components
            and len(dev_indices) + len(component_list[component_index]) <= target
            and all(remaining[category] > 1 for category in component_categories[component_index])
        )

    def select(component_index: int) -> None:
        selected_components.add(component_index)
        dev_indices.update(component_list[component_index])
        dev_counts.update(component_counts[component_index])
        for category in component_categories[component_index]:
            remaining[category] -= 1

    # Reserve one component for every category with more than one component.
    for category in sorted(by_category, key=lambda name: (len(by_category[name]), name)):
        if len(by_category[category]) < 2:
            continue
        if any(index in selected_components for index in by_category[category]):
            continue
        candidates = sorted(
            by_category[category],
            key=lambda index: (len(component_list[index]), order(component_list[index])),
        )
        for component_index in candidates:
            if can_select(component_index):
                select(component_index)
                break

    # Fill toward the original category mix, subject to indivisible groups.
    while True:
        candidates = [index for index in range(len(component_list)) if can_select(index)]
        if not candidates:
            break

        def improvement(component_index: int) -> float:
            gain = 0.0
            for category, count in component_counts[component_index].items():
                desired = category_targets[category]
                before = (dev_counts[category] - desired) ** 2 / desired
                after = (dev_counts[category] + count - desired) ** 2 / desired
                gain += before - after
            return gain / len(component_list[component_index])

        best = min(candidates, key=lambda index: (-improvement(index), order(component_list[index])))
        select(best)

    train = [row for index, row in enumerate(rows) if index not in dev_indices]
    dev = [row for index, row in enumerate(rows) if index in dev_indices]
    return train, dev


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    with SOURCE.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("The source CSV has no header")
        train_rows, dev_rows = split(list(reader))
    write_csv(TRAIN, train_rows, fields)
    write_csv(DEV, dev_rows, fields)
    print(f"Created {TRAIN} ({len(train_rows)} rows) and {DEV} ({len(dev_rows)} rows)")
