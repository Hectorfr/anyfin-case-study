# Anyfin KTH case night — Track A starter
Run: `python predict.py data/validation.csv`; output: `predictions.csv` with `id,category`.
Approach: conservative rules first, then train-only merchant matching; uncertain rows are `unknown`.
Cost at 10 million rows/month: $0 in model/API fees; compute and memory costs remain unmeasured.
With another day: add a calibrated model fallback for genuinely unseen businesses.
