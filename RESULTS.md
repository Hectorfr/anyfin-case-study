# Track A merchant matching results

The fixed development set contains 138 labelled rows. The merchant index uses only `data/train.csv`; the official unlabelled `data/validation.csv` was not used for these comparisons. The baseline rules were developed after inspecting the original full ledger. Payout-marker and name-extension handling were refined after the first development run showed no gain. These scores show progress during development, not an independent accuracy estimate.

| Configuration | Correct | Wrong | Unknown | Score per 100 | Precision | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules only | 35 | 0 | 103 | 25.36 | 100% | 25.36% |
| Rules + exact merchant match | 39 | 0 | 99 | 28.26 | 100% | 28.26% |
| Rules + exact, extension, chain and fuzzy match | 41 | 0 | 97 | 29.71 | 100% | 29.71% |

The six additional correct rows are four gambling payouts matched exactly after removing the payout marker and two transport purchases matched to a known merchant with a location suffix. The gains were two rows at Nordea and four at Swedbank; Handelsbanken was unchanged. No development row was newly labelled by fuzzy or chain matching, so their usefulness on real new spelling variants remains unmeasured. Train-only grouped checks scored 80 correct and 0 wrong with rules alone, versus 121 correct and 0 wrong with the full matcher, across 558 rows. A synthetic one-character deletion check recovered all 57 generated merchant variants at the chosen 0.88 similarity threshold and 0.10 different-category margin: 44 through fuzzy matching and 13 through chain matching. That check does not establish performance on genuinely unseen businesses.

Reproduce the development comparison:

```sh
python predict.py data/dev.csv --matching none --output predictions/dev-rules.csv
python score.py data/dev.csv predictions/dev-rules.csv
python predict.py data/dev.csv --matching exact --output predictions/dev-exact.csv
python score.py data/dev.csv predictions/dev-exact.csv
python predict.py data/dev.csv --matching full --output predictions/dev-full.csv
python score.py data/dev.csv predictions/dev-full.csv
```

The full 138-row run took about 0.02 seconds on this machine and makes no API calls. That tiny runtime sample is not a 10-million-row capacity estimate. The input is loaded in memory to pair transfers, so production-scale processing would require a different batching or pairing strategy.

## Optional local text classifier

The optional character n-gram TF-IDF and logistic model trains on all 558 `data/train.csv` rows and runs only after rules and merchant matching return `unknown`. A three-fold split by normalized description selected the 0.60 raw model-score threshold on train data (14 added correct, 0 wrong across 393 fallback rows). The raw score is **not** calibrated as a probability.

| Configuration | Correct | Wrong | Unknown | Score per 100 | Precision | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules + merchant matcher | 41 | 0 | 97 | 29.71 | 100% | 29.71% |
| Plus optional text model | 48 | 1 | 89 | **34.06** | 97.96% | 35.51% |

The added results are seven correct and one wrong. Six correct rows are repeat purchases from one business, so the gain spans few distinct businesses. The wrong row is a leisure purchase classified as groceries after a location word matched grocery examples. This dev result is exploratory rather than independent evidence for future customers.

Run the optional comparison after installing `requirements-model.txt`:

```sh
python evaluate_text_model.py
python predict.py data/dev.csv --text-model --output predictions/dev-text.csv
python score.py data/dev.csv predictions/dev-text.csv
```

A synthetic 5,517-row input with unique description suffixes completed in about 1 second on this machine, including model fitting, and produced 5,517 output rows. It does not measure accuracy or a production-scale memory footprint. There are no model API fees; `scikit-learn` is an optional local dependency.
