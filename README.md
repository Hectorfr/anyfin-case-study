# Anyfin KTH case night — Track A starter
Install: `pip install -r requirements-model.txt`; run: `python predict.py data/validation.csv --full-training --text-model --output solutions.csv`.
Approach: rules, merchant matching and local text model fitted on all 696 labelled rows; uncertain rows are `unknown`; output is `id,category`.
Cost at 10 million rows/month: $0 in model/API fees; compute and memory costs remain unmeasured.
With another day: calibrate the text model on more customers and investigate genuinely unseen businesses.
