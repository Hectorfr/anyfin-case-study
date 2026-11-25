# Anyfin Track A transaction classifier
Approach: rules, merchant matching, character n-gram model, then batched Claude Haiku 5.5 for well-supported remaining labels; otherwise `unknown`.
Run: `pip install -r requirements-model.txt`; set `ANTHROPIC_API_KEY` in `.env` (copy `.env.example`); `python predict.py INPUT.csv --full-training --text-model --claude --output predictions.csv`.
Cost at 10 million rows/month: about $130 in Claude tokens at the validation set's description mix, plus compute; raise the current 2,000-group cap to keep the same coverage.
With another day: cache Claude decisions across runs, parallelize API batches within rate limits, and pair transfers with bounded memory.
