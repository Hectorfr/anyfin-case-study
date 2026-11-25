# Anyfin KTH case night — Track A starter
Run: `python predict.py data/validation.csv` (optional `--text-model` after `pip install -r requirements-model.txt`); output: `predictions.csv` with `id,category`.
Approach: rules, train-only merchant matching, then an optional local text model; uncertain rows are `unknown`.
Cost at 10 million rows/month: $0 in model/API fees; compute and memory costs remain unmeasured.
With another day: calibrate the text model on more customers and investigate genuinely unseen businesses.
