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

## Claude fallback on unseen descriptions

The optional `--claude` stage runs only for rows still `unknown` after rules, merchant matching and the local text model. It batches unique descriptions, uses Claude Haiku 5.5's structured JSON output, and accepts a label only when Claude marks it high certainty, names a recognizable business or explicit purpose, and quotes supporting text actually present in the transaction. This is a conservative gate, not a calibrated probability. The prompt and gate were fixed before the first Claude development run.

| Configuration on fixed dev | Correct | Wrong | Unknown | Score per 100 | Precision | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules + matcher + local text | 48 | 1 | 89 | 34.06 | 97.96% | 35.51% |
| Plus Claude fallback | 127 | 1 | 10 | **91.30** | 99.22% | 92.75% |

With the final response limit, Claude added 79 correct and 0 wrong labels across 38 distinct description/context groups. The one wrong label in the combined pipeline came from the preceding local text model: a Nordea `SF BIO SODERMALM` cinema purchase was called groceries. The first, smaller response limit produced 84.78 on dev and truncated a validation batch, so the final limit was raised. Two runs of the final configuration produced identical dev predictions, though output-token usage differed. The dev set is small and was already used while developing previous stages, so this is exploratory evidence, not a guaranteed hidden-set score. Each final dev run made 3 API requests using 7,831 input and about 2,300 output tokens.

Reproduce after putting `ANTHROPIC_API_KEY` in the ignored `.env` file:

```sh
python predict.py data/dev.csv --text-model --claude --output predictions/dev-claude.csv
python score.py data/dev.csv predictions/dev-claude.csv
```

Claude Haiku 5.5 is listed at $0.10 per million input tokens and $0.50 per million output tokens for prompts under 100,000 tokens. At those rates, each successful final dev run's token use costs about $0.002 and the successful validation run's use costs about $0.016, excluding earlier retries. The validation run used 16 requests, 47,527 input and 21,928 output tokens, and took about 75 seconds here. If hidden data has a similar mix of unique descriptions, 5,517 rows would scale to roughly six minutes and under $0.10 of model tokens; actual rate limits and runtime need a hidden-run check. A 10-million-row month at the same mix would imply about $130 of model tokens if the 2,000-group cap were raised; batching, parallelism and persistent caching would be needed for that volume.

API reference: [Claude Messages](https://platform.claude.com/docs/en/api/messages/create), [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Haiku 5.5 pricing](https://platform.claude.com/docs/en/models/haiku-5-5/overview).

## Full-training validation predictions

After development, `--full-training` fits both merchant matching and the optional text model on all 696 labelled rows in `data/transactions.csv` instead of the 558-row training split. The generated [solutions.csv](solutions.csv) also uses the Claude fallback:

```sh
python predict.py data/validation.csv --full-training --text-model --claude --output solutions.csv
```

The file has exactly 1,183 unique validation IDs in source order and only valid `id,category` values. It labels 1,033 rows and returns `unknown` for 150 (87.32% coverage). Claude labelled 519 rows that the full-trained local pipeline left `unknown`, without changing any existing local label. The validation file has no gold labels, so these counts do not establish its score. The 91.30 development score above applies only to the split-trained model; scoring the full-trained model on dev would leak its training labels.
