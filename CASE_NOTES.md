# Track A: problem and possible approaches

Source: [Anyfin × KTH AI Society case night](https://anyfin.github.io/kth-case-night/#label), 25 November 2026. The supplied transactions are synthetic. This repository contains the public training and validation CSVs for convenience; it does not contain event API keys.

The next steps and comparison gates are in [PLAN.md](PLAN.md).
The fixed development results are in [RESULTS.md](RESULTS.md).

## Task and constraints

- Predict exactly one of 19 transaction categories or `unknown` for each row. Input has `id,customer,bank,account,account_name,account_type,date,amount_sek,text`; output is exactly `id,category`.
- Official training: 696 labelled rows, three customers, three bank text styles. The official `validation.csv` is treated as our unlabelled test input: 1,183 rows from five new customers, mostly unseen merchants and a fourth bank style. The organizers also have a hidden final set of 5,517 rows from 24 further customers.
- Score per 100 rows is `100 × (correct − wrong) / total`. `unknown` scores zero. Accuracy counts correct over all rows; precision counts correct over labelled rows; coverage is the share labelled. Missing predictions count as `unknown`.
- The submission must reproduce predictions from code with one command. Hidden test execution targets under 10 minutes and $5 for 5,517 rows. The short README should state the approach, projected cost for 10 million rows/month, next steps, and command.

## What makes it hard

The test shifts both customer and merchant distributions. Bank text has dates, prefixes, payment processors, account transfers and ambiguous merchant names. Amount sign alone does not determine class: refunds and payouts keep the original purchase category. A savings transfer appears on both accounts, and the bank's nominal account type may not reflect its use. Credit card bill payments differ from the matching receipt on the card account. Payment provider invoices differ from card purchases routed through those providers. A person-to-person transfer with a stated purpose takes that purpose's category.

## Current rules and merchant matcher

`predict.py` recognizes explicit salary text, matched transfers between a customer's accounts, savings-account transfers, credit-card bill receipts, cash withdrawals, and a narrow set of loan and collections payment markers. Everything else is `unknown`. On the 696 labelled training rows it yielded 127 correct, 0 wrong, 569 unknown: score 18.25, precision 100%, coverage 18.25%. This is an in-sample diagnostic after inspecting the training data, not a generalization estimate. It produced one prediction per validation ID; the validation file has no labels, so its accuracy is unknown. The script loads the input into memory to match transfers; a 10-million-row production batch would require chunking or an external pairing index.

`merchant_matching.py` now builds a spending-merchant index from `data/train.csv` and runs only after a rule returns `unknown`. It strips bank and payment-provider wrappers, then attempts exact, safe name-extension, consistent chain, and conservative fuzzy matches. On the fixed development set, rules plus matching scored 29.71 versus 25.36 for rules alone, with zero wrong labels in either run. Most development businesses remain unseen in training, so this stage has limited coverage; a model fallback is the next stage.

An optional local character n-gram logistic classifier now runs after the matcher with `--text-model`. It requires `requirements-model.txt` but makes no API calls. The 0.60 score threshold was selected on train-only description-group folds. On dev it added seven correct labels and one wrong label, raising the score to 34.06; six of those correct rows share one merchant. Its score is not a calibrated probability, so the default command still stops after merchant matching.

For final inference, `--full-training` fits both the matcher and optional text classifier on all 696 labelled transactions. `solutions.csv` was generated from the 1,183 unlabelled validation rows with both flags; 514 received a category and 669 received `unknown`. This is coverage only, since validation labels are unavailable.

## Approaches to investigate later

1. **Extend high-certainty rules:** Normalize more bank wrappers and merchant strings. Keep provenance for every rule.
2. **Merchant knowledge and fuzzy matching:** Build a small merchant/category lookup or retrieval index. Evaluate on held-out merchants to avoid memorizing the training set.
3. **LLM fallback:** Ask a model for ambiguous rows in batches using the full allowed context and taxonomy. Cache repeated normalized descriptions. Measure latency and cost; keep keys in environment variables.
4. **Selective decision:** Estimate confidence and emit `unknown` when a label is more likely wrong than right. At +1/−1/0, predicting has positive expected value only above 50% correctness; calibration and actual score matter more than raw coverage.

## Evaluation plan

`data/train.csv` (558 rows) and `data/dev.csv` (138 rows) are a fixed split of the 696 labelled rows, generated by `python make_split.py`. The 20% development target keeps normalized repeated descriptions together and keeps both entries of a matched account transfer together. It also reserves development examples for categories with more than one description group and roughly balances the remaining category mix. This tests unseen descriptions better than a random row split, but all three customers and bank formats appear on both sides, and four categories with only one description group are absent from dev. The rules above were already designed after inspecting the full source, so the initial dev result of 35 correct, 0 wrong and 103 unknown (score 25.36) is a comparison baseline rather than an unbiased estimate of generalization. For future trained models, fit only `data/train.csv` while comparing runs on `data/dev.csv`.

To compare a run:

```sh
python predict.py data/dev.csv --output predictions/dev-baseline.csv
python score.py data/dev.csv predictions/dev-baseline.csv
```

Keep `data/validation.csv` as an unlabelled test input until choosing a final approach. It can check output shape, coverage and runtime, but not local accuracy. The official hidden set remains the ultimate benchmark.

## Data provenance

`data/transactions.csv` and `data/validation.csv` are copies of the public [Anyfin case repository](https://github.com/anyfin/kth-case-night), accessed 25 November 2026. `data/train.csv` and `data/dev.csv` are derived from `data/transactions.csv`. They contain synthetic customer data. The `event-key.json` and `dev-key.json` files in that repository are intentionally excluded.
