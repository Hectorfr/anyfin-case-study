# Anyfin KTH case night — Track A starter
Run: `python predict.py data/validation.csv`; output: `predictions.csv` with `id,category`.
Approach: conservative rules for salary, own-account transfers, cash withdrawals and debt payments; all other rows are `unknown`.
Cost at 10 million rows/month: $0 in model/API fees; compute and memory costs remain unmeasured.
With another day: test merchant matching on held-out customers and add a confidence-based fallback.
