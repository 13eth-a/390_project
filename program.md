# Program Specification — Modeling Political Ad Messaging Effectiveness

**Course:** POLI/STAT 390 Capstone  
**Last Updated:** April 29, 2026  
**Status:** Week 2 — Baseline Established

---

## 1. Project Goal

Predict the relative effectiveness of political Facebook advertisements using only their textual content (and lightweight metadata). "Effectiveness" is operationalized as the midpoint of the ad's reported impressions range, log-transformed. The primary research question is:

> *To what extent do the words and framing of a political ad predict how widely it was distributed?*

This is framed as a regression/ranking problem rather than a causal inference problem. The pipeline is designed to be reproducible and extendable to automated hyperparameter search in later weeks.

---

## 2. Data

| Item | Detail |
|------|--------|
| Source | Facebook Political Ads (ProPublica FBPAC dataset) |
| Raw file | `data/raw/fbpac-ads-en-US.csv` |
| Processed file | `data/processed/political_ads_processed.csv` |
| Time filter | 2022–2023 (configurable in `run_baseline.py`) |
| Train / Val / Test split | 70 / 15 / 15 (hash-based, deterministic) |
| n_train (current) | 113,698 |
| n_val (current) | 24,296 |

**Planned upgrade:** In later weeks the dataset will be replaced with a direct Meta Ad Library export (2022–2024) to include more recent advertising cycles.

**Features used:**
- Ad body text (cleaned: lowercased, URLs removed, whitespace normalized)
- Sponsor / page name (not yet used in model, available in outputs)
- State / region (not yet used in model)

**Target:** `log_impressions_mid = log(1 + impressions_midpoint)`

---

## 3. Modeling Pipeline

```
Load raw CSV
  → infer column names (flexible schema)
  → parse impressions range → midpoint
  → clean and filter text
  → hash-based train/val/test split
  → save processed CSV

Load processed CSV
  → TF-IDF vectorization (fit on train only)
  → Ridge Regression (fit on train)
  → Predict on validation set
  → Compute Spearman ρ and RMSE
  → Save metrics.json and val_predictions.csv
```

### 3.1 Baseline Configuration (Frozen Week 2)

| Parameter | Value |
|-----------|-------|
| Vectorizer | TF-IDF |
| `max_features` | 5,000 |
| `ngram_range` | (1, 2) |
| `min_df` | 2 |
| `max_df` | 0.95 |
| Model | Ridge Regression |
| `alpha` | 1.0 |
| `random_state` | 42 |

### 3.2 Evaluation Metrics (Frozen)

- **Primary:** Spearman Rank Correlation (val set) — measures ranking quality
- **Secondary:** RMSE on log-impressions — measures absolute prediction error

Metrics format (never changed after Week 2):

```json
{
  "spearman": <float>,
  "rmse": <float>,
  "runtime_seconds": <float>
}
```

---

## 4. Key Constraints

1. **Test set is held out.** It must not be used until final evaluation. All tuning uses the validation set only.
2. **Evaluation format is frozen.** Metric names and JSON schema must not change between weeks.
3. **Splits are deterministic.** Hash-based assignment ensures the same rows always land in the same split, even if the CSV is reordered.
4. **No data leakage.** The TF-IDF vectorizer is fit exclusively on the train split; the vocab is then applied to val/test.
5. **Reproducibility.** All models use `random_state=42`. The pipeline must produce identical outputs on repeated runs.

---

## 5. Week-by-Week Roadmap

| Week | Objective | Status |
|------|-----------|--------|
| 1 | Project scoping, dataset acquisition, environment setup | Done |
| 2 | Frozen baseline pipeline, evaluation framework, initial experiments | **Current** |
| 3 | Feature engineering (sponsor embeddings, temporal features, geo features) | Planned |
| 4 | Automated hyperparameter search (Optuna or grid search over TF-IDF + model params) | Planned |
| 5 | Alternative models (Gradient Boosting, LightGBM, BERT embeddings) | Planned |
| 6 | Error analysis, ablation studies, final test-set evaluation | Planned |
| 7 | Final report and presentation | Planned |

---

## 6. File Structure

```
390_project/
├── program.md               ← this file
├── research_log.md          ← experiment logs and reflections
├── README.md                ← setup and reproducibility instructions
├── run_baseline.py          ← entry point for the default pipeline run
├── requirements.txt
├── src/
│   ├── data_loader.py       ← schema-agnostic data loading and splitting
│   └── baseline_pipeline.py ← TF-IDF + Ridge training and evaluation
├── data/
│   ├── raw/                 ← (gitignored) raw source CSVs
│   └── processed/           ← (gitignored) cleaned, split datasets
└── artifacts/
    └── baseline/            ← metrics.json, val_predictions.csv
```

---

## 7. Success Criteria

| Milestone | Target |
|-----------|--------|
| Baseline Spearman (val) | ≥ 0.30 ✓ (achieved 0.333) |
| Week 3 Spearman (val) | ≥ 0.40 |
| Final Spearman (test) | ≥ 0.45 |
| Reproducibility | Identical metrics across ≥ 3 independent runs |
| Runtime per experiment | < 120s |

---

## 8. Agent / Automation Plan (Week 3+)

In later weeks an LLM-based coding agent (GitHub Copilot) will be used to:
- Propose and implement new feature engineering functions
- Generate hyperparameter search configurations
- Summarize experiment results and suggest next steps

The agent will be evaluated on:
- Whether its code changes run without modification
- Whether proposed changes improve val Spearman
- Whether it respects the frozen evaluation contract (no test-set peeking, no metric schema changes)
