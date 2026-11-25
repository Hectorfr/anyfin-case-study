# Track A transaction classifier: current plan

The objective is to maximize `100 × (correct − wrong) / rows`; `unknown` scores zero. The current submission candidate is a four-stage classifier in `predict.py`. It produces labels from code for every input row and abstains when none of the stages has enough evidence.

## Implemented pipeline

1. **Rules:** Identify explicit salary, matched transfers between a customer's accounts, savings movements, card-bill receipts, cash withdrawals, and narrow loan and collections payments.
2. **Merchant matching:** Remove bank and payment-provider wrappers, then use exact, name-extension, chain and conservative fuzzy matches against labelled training merchants. Conflicting merchant labels abstain.
3. **Local text model (`--text-model`):** Character n-gram TF-IDF and logistic regression handle rows still `unknown`. A 0.60 raw-score cutoff was selected on train-only description-group folds. Its score is not a calibrated probability.
4. **Claude (`--claude`):** Claude Haiku 5.5 receives only rows still `unknown`. It groups repeated descriptions, sends batches of 20, and caps each run at 2,000 unique groups. A label is accepted only when the structured response says `high` certainty, identifies a known business or explicit purpose, and quotes an exact supporting span from the transaction text. Otherwise the row remains `unknown`. Claude's certainty is not a calibrated probability.

The input rows pass through the local stages in order; the residual rows are then batched for Claude. The Claude key comes from `ANTHROPIC_API_KEY` in the environment or the ignored `.env` file described in `.env.example`. Never commit a real key.

## Data and comparisons

- **Development:** `data/train.csv` has 558 labelled rows and `data/dev.csv` has 138. Keep this split fixed for comparisons. Do not use `--full-training` when scoring dev: that would put dev labels into the merchant index and local model.
- **Final fitting:** `--full-training` uses all 696 labelled rows in `data/transactions.csv` for merchant matching and the local text model. Claude is prompted with the fixed taxonomy, not trained on these rows.
- **Unlabelled inputs:** `data/validation.csv` has 1,183 rows; `data/final_test.csv` has 800. Neither has answer labels, so coverage and output validity can be checked, but score and accuracy cannot.

| Fixed dev configuration | Correct | Wrong | Unknown | Score per 100 |
| --- | ---: | ---: | ---: | ---: |
| Rules | 35 | 0 | 103 | 25.36 |
| Rules + merchant matching | 41 | 0 | 97 | 29.71 |
| Plus local text model | 48 | 1 | 89 | 34.06 |
| Plus Claude, final response limit | 127 | 1 | 10 | **91.30** |

The final Claude setup added 79 correct and no wrong labels on dev in two runs with identical predictions. The remaining wrong label came from the local text model. This split is small, repeats some merchants, and earlier rules were developed after inspecting the original ledger; the score is development evidence, not a reliable estimate for new customers. Detailed measurements and costs are in [RESULTS.md](RESULTS.md).

## Current outputs and reproduction

Install `requirements-model.txt` and set `ANTHROPIC_API_KEY` in the ignored `.env` file or shell. Then run:

```sh
python predict.py data/dev.csv --text-model --claude --output predictions/dev-claude.csv
python score.py data/dev.csv predictions/dev-claude.csv
python predict.py data/validation.csv --full-training --text-model --claude --output solutions.csv
python predict.py data/final_test.csv --full-training --text-model --claude --output final_solution.csv
```

`solutions.csv` contains 1,183 IDs, with 1,033 labels and 150 `unknown`. The local `final_solution.csv` contains 800 IDs, with 715 labels and 85 `unknown`. Both were checked for exact input ID order, unique IDs, and valid `id,category` output. `data/final_test.csv` and `final_solution.csv` are currently local untracked files. No validation or final-test accuracy is known.

## Next work

- [ ] Obtain an independent labelled sample from new customers or banks. Measure score, precision and coverage by source, category, bank and distinct merchant. Keep the unlabelled files out of threshold tuning.
- [ ] Calibrate the local model's 0.60 cutoff and Claude acceptance gate on held-out merchant or customer groups. A predicted label must have a credible chance above 50% to beat `unknown` under this scoring rule; use a safety margin for shifts.
- [ ] Benchmark the 5,517-row hidden-run shape against the under-10-minute and under-$5 limits. The 1,183-row validation run took about 75 seconds and used 16 Claude requests; a roughly six-minute hidden estimate assumes a similar mix and rate limits.
- [ ] For production volume, use persistent caching, parallel batches, and bounded-memory transfer pairing. The current 2,000-group Claude cap and in-memory input loading need changes before processing 10 million rows in one run.
- [ ] Rotate the Claude key that was shared in chat, update the ignored `.env`, and confirm the five-line README and final command before submission.

Case source: [Anyfin × KTH AI Society case night](https://anyfin.github.io/kth-case-night/#label).
