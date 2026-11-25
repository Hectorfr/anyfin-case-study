# Anyfin KTH case night — Track A starter
Setup/run: `pip install -r requirements-model.txt`; put `ANTHROPIC_API_KEY` in ignored `.env` (see `.env.example`); `python predict.py data/validation.csv --full-training --text-model --claude --output solutions.csv`.
Approach: rules, merchant matching, local text model, then batched Claude Haiku 5.5 for high-evidence unknowns; output is `id,category`.
Estimated Claude cost at 10 million validation-like rows/month: about $130, excluding compute; the 2,000-group run cap needs adjustment at that scale.
With another day: validate on more customers, calibrate abstention and benchmark production-scale throughput.
