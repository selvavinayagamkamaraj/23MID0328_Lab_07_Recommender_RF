# MDI3003 — Experiment 07: Recommendation System with Random Forest

**Name:** Selva vinayagam kamaraj  **Registration No.:** 23MID0328
**Course:** MDI3003 — Advanced Predictive Analytics, Dr. Durgesh Kumar, SCOPE, VIT Vellore
**Dataset:** UCI Online Retail (D1) — 541,909 real transactions, 01-Dec-2010 to 09-Dec-2011.

## What this is

An end-to-end, leakage-safe Random Forest recommendation pipeline: chronological split →
bounded/recall-audited candidate catalog → customer/item/pair feature engineering →
training-only hyperparameter selection → one-shot locked-test evaluation → Top-K ranking →
five-case error/cold-start audit. Full narrative, all required figures, and every result
table are in the report; the notebook reproduces every number in it from raw data.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/
│   └── onlineretail.csv                     # raw UCI Online Retail extract (semicolon-delimited, European decimals)
├── 23MID0328_Lab07_Recommender_RF.ipynb      # full pipeline, runs top to bottom
├── 23MID0328_Lab07_Report.pdf                # full report (16 pages, grayscale figures)
├── outputs/
│   ├── 23MID0328_Lab07_Ranking_Metrics.csv   # Popularity vs RF Precision/Recall/HitRate@5,10
│   ├── 23MID0328_Lab07_Candidate_Recall.csv  # candidate-recall audit, both windows
│   ├── 23MID0328_Lab07_Recommendations.csv   # Top-10 RF recommendations, all warm test customers
│   └── 23MID0328_Lab07_Error_Analysis.csv    # per-customer hit counts behind the five-case audit
├── artifacts/
│   ├── dataset_card.json                     # source, hash, cleaning counts
│   ├── split_manifest.json                   # frozen chronological cutoffs
│   ├── candidate_policy.json                 # frozen 1,500-item candidate catalog
│   ├── feature_schema.json                   # ordered list of the 19 model features
│   ├── hyperparam_validation.csv             # 3-config tuning comparison
│   ├── feature_importance.csv                # Gini importance, final model
│   └── feature_ablation.csv                  # pair-history ablation result
├── figures/                                  # all 10 required grayscale PNGs
└── models/                                   # random_forest.joblib written here on first run (not committed — see below)
```

## Reproducing the results

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook 23MID0328_Lab07_Recommender_RF.ipynb
```

Run all cells top to bottom. `data/onlineretail.csv` is already included, so no download step
is required. The notebook writes `models/random_forest.joblib`, refreshed copies of every file
under `artifacts/` and `outputs/`, and PNGs under `figures/`.

Random seed is fixed at 42 throughout (numpy and scikit-learn); re-running should reproduce the
report's numbers exactly.

## Why the model file isn't committed

The final Random Forest (`n_estimators=300`, unrestricted `max_depth`) serializes to ~684 MB —
over GitHub's 100 MB hard limit — so `models/*.joblib` is git-ignored. Regenerate it in under
three minutes by running the notebook, or track it with Git LFS if you need it versioned.

## Headline result

Locked test window, warm customers (n=1,322): Random Forest reaches **Precision@10 = 0.633**,
**Recall@10 = 0.372**, **HitRate@10 = 0.988**, versus the popularity baseline's 0.075 / 0.035 /
0.474 — roughly an 8x precision gain. Candidate recall is ~0.78 on the same window, reported
explicitly as the ceiling on achievable Recall@K (see `outputs/23MID0328_Lab07_Candidate_Recall.csv`
and Section 6 of the report). Full breakdown, K-sensitivity, feature importance, ablation, and
the five-case/cold-start audit are in the report.

## Known limitations (documented in the report, Section 16)

- Only 3 of 4 planned hyperparameter configs and 1 of 3 planned feature ablations were run
  within the lab's 3-hour time-box.
- No advanced-learner benchmark (collaborative filtering, matrix factorization, neural models)
  is included — flagged as future work, not silently skipped.
- Negative-sampling ratio sensitivity (E5) and cross-dataset replication (E9) were not tested.
