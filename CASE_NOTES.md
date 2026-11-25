# Track A: problem and possible approaches

Source: [Anyfin × KTH AI Society case night](https://anyfin.github.io/kth-case-night/#label), 25 November 2026. The supplied transactions are synthetic. This repository contains the public training and validation CSVs for convenience; it does not contain event API keys.

## Task and constraints

- Predict exactly one of 19 transaction categories or `unknown` for each row. Input has `id,customer,bank,account,account_name,account_type,date,amount_sek,text`; output is exactly `id,category`.
- Training: 696 labelled rows, three customers, three bank text styles. Validation: 1,183 unlabelled rows, five new customers, mostly unseen merchants and a fourth bank style. Hidden test: 5,517 rows from 24 further customers.
- Score per 100 rows is `100 × (correct − wrong) / total`. `unknown` scores zero. Accuracy counts correct over all rows; precision counts correct over labelled rows; coverage is the share labelled. Missing predictions count as `unknown`.
- The submission must reproduce predictions from code with one command. Hidden test execution targets under 10 minutes and $5 for 5,517 rows. The short README should state the approach, projected cost for 10 million rows/month, next steps, and command.

## What makes it hard

The test shifts both customer and merchant distributions. Bank text has dates, prefixes, payment processors, account transfers and ambiguous merchant names. Amount sign alone does not determine class: refunds and payouts keep the original purchase category. A savings transfer appears on both accounts, and the bank's nominal account type may not reflect its use. Credit card bill payments differ from the matching receipt on the card account. Payment provider invoices differ from card purchases routed through those providers. A person-to-person transfer with a stated purpose takes that purpose's category.

## Current first-pass baseline

`predict.py` recognizes explicit salary text, matched transfers between a customer's accounts, savings-account transfers, credit-card bill receipts, cash withdrawals, and a narrow set of loan and collections payment markers. Everything else is `unknown`. On the 696 labelled training rows it yielded 127 correct, 0 wrong, 569 unknown: score 18.25, precision 100%, coverage 18.25%. This is an in-sample diagnostic after inspecting the training data, not a generalization estimate. It produced one prediction per validation ID; the validation file has no labels, so its accuracy is unknown. The script loads the input into memory to match transfers; a 10-million-row production batch would require chunking or an external pairing index.

## Approaches to investigate later

1. **Extend high-certainty rules:** Normalize more bank wrappers and merchant strings. Keep provenance for every rule.
2. **Merchant knowledge and fuzzy matching:** Build a small merchant/category lookup or retrieval index. Evaluate on held-out merchants to avoid memorizing the training set.
3. **LLM fallback:** Ask a model for ambiguous rows in batches using the full allowed context and taxonomy. Cache repeated normalized descriptions. Measure latency and cost; keep keys in environment variables.
4. **Selective decision:** Estimate confidence and emit `unknown` when a label is more likely wrong than right. At +1/−1/0, predicting has positive expected value only above 50% correctness; calibration and actual score matter more than raw coverage.

## Evaluation plan

Split the 696 labelled rows by customer for a rough development check, and separately hold out merchants or normalized merchant groups. Avoid random row splits: repeated salary, rent and merchant strings can inflate a score. Track score, precision, coverage and error examples by category and bank. The public validation set has no labels, so it checks output shape and runtime but cannot yield a local accuracy estimate. Reserve time to run the exact submission command and inspect that IDs are complete and unique. The official page's training scorer can check predictions on the labelled data.

## Data provenance

`data/transactions.csv` and `data/validation.csv` are copies of the public [Anyfin case repository](https://github.com/anyfin/kth-case-night), accessed 25 November 2026. They contain synthetic customer data. The `event-key.json` and `dev-key.json` files in that repository are intentionally excluded.
