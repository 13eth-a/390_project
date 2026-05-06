# Failure Analysis Memo
**Project:** POLI/STAT 390 — Modeling Political Ad Messaging Effectiveness
**Date:** May 6, 2026
**Author:** Research Log (auto-generated from experiment results)

---

## 1. How the Experiment Was Controlled

All experiments share a **single frozen evaluation protocol**:

| Control dimension | Fixed value |
|-------------------|-------------|
| Dataset | `data/processed/political_ads_processed.csv` |
| Train / Val / Test split | 70 / 15 / 15, hash-based, deterministic |
| Target variable | `log_impressions_mid = log(1 + impressions midpoint)` |
| Evaluation set | Validation set only (test set untouched) |
| Metrics | Spearman ρ (primary), RMSE (secondary) |
| Random seed | 42 on all sklearn estimators |
| n_train | 113,698 ads |
| n_val | 24,296 ads |

Each experiment changes **exactly one axis** relative to the frozen baseline
(TF-IDF 5k vocab, bigrams, Ridge α=1.0):

- **Naive baselines** (B1, B2): no text model at all → establish floor
- **Vocab sweep** (C1–C6): only `max_features` varies (1k → 50k)
- **N-gram variation** (C7, Exp-02): only `ngram_range` varies
- **Regularisation variation** (Exp-03, Exp-04): only Ridge `alpha` varies

All other hyperparameters are held at baseline values within each series.

---

## 2. Error Taxonomy

### Tier 1 — Model Capacity Errors
Errors arising because the model lacks sufficient representational power
for the task.

| Code | Name | Description | Experiments affected |
|------|------|-------------|----------------------|
| MC-1 | Vocabulary truncation | TF-IDF vocabulary too small; frequent political bigrams dropped | C1, C2, C3 |
| MC-2 | Unigram poverty | Restricting to unigrams loses phrase-level signal (e.g. "vote for", "fight back") | Exp-02 |
| MC-3 | Linear ceiling | Ridge can only capture linear word-impression relationships; non-linear patterns invisible | All TF-IDF exps |
| MC-4 | Sparse representation | TF-IDF bag-of-words discards word order and semantic similarity | All TF-IDF exps |

### Tier 2 — Data Quality Errors
Errors arising from the structure or coverage of the training data.

| Code | Name | Description | Experiments affected |
|------|------|-------------|----------------------|
| DQ-1 | Missing dates | `delivery_start` is NaN for all rows; temporal features unavailable | All exps |
| DQ-2 | URL-format sponsors | Sponsor field contains FB URLs, not page names; join with advertiser report fails | B2, all |
| DQ-3 | Era mismatch | FBPAC dataset reflects 2018–2020 ad landscape, not 2022–2023 | All exps |
| DQ-4 | Impressions range coarseness | Target derived from midpoint of broad ranges (e.g. "100K–200K"); introduces systematic label noise | All exps |

### Tier 3 — Evaluation Design Errors
Errors in the measurement framework itself.

| Code | Name | Description | Experiments affected |
|------|------|-------------|----------------------|
| ED-1 | No confidence intervals | Single val-set Spearman cannot distinguish real gains from noise; Δρ < 0.01 is meaningless | All exps |
| ED-2 | Spearman on noisy labels | Spearman measures ranking quality against noisy midpoints, not true reach | All exps |
| ED-3 | No test-set confirmation | All comparisons are val-set only; best config not yet confirmed out-of-sample | All exps |

### Tier 4 — Operational / Pipeline Errors
Errors encountered in the data collection and infrastructure layer.

| Code | Name | Description | Status |
|------|------|-------------|--------|
| OP-1 | Meta API access blocked | Business verification required for `/ads_archive`; no academic bypass | Unresolved |
| OP-2 | App token insufficient | App-level tokens return Error 10 for political ad endpoints | Resolved (user token) |
| OP-3 | Token scope mismatch | `ads_read` is a Marketing API permission, not a Login permission | Resolved |
| OP-4 | Legacy dataset dates absent | FBPAC raw CSV dates did not parse; `delivery_start` all NaN | Unresolved |

---

## 3. Key Findings

### 3.1 Naive Baselines Reveal Weak Signal

The sponsor-mean baseline (B2) achieves Spearman **0.325** using
only the advertiser identity — no text at all. The frozen TF-IDF baseline
achieves **0.333**, only **0.008** above the
no-text baseline. This means:

> **Roughly 97% of the ranking signal the model captures could be
> explained by "who placed the ad" rather than "what the ad says."**

This is the most important finding of the experiment set. It suggests
either (a) the text-only approach has a hard performance ceiling, or
(b) advertiser identity needs to be incorporated as a feature alongside
text before the text signal can be disentangled.

### 3.2 Vocabulary Size Is the Highest-Value Lever

The vocab sweep shows a clean monotonic relationship:

| Vocab | Spearman | RMSE |
|-------|----------|------|
| 1k    | 0.267    | 0.629 |
| 2.5k  | 0.299    | 0.621 |
| 5k    | 0.333    | 0.611 |
| 10k   | 0.350    | 0.605 |
| 25k   | 0.376    | 0.596 |
| 50k   | **0.390**    | **0.589** |

The curve has not yet plateaued at 50k. The next experiment should test
100k and the full vocabulary.

### 3.3 Regularisation and N-gram Changes Produce Noise-Level Effects

All alpha and n-gram variants changed Spearman by less than ±0.005 relative
to the 5k baseline. Given the absence of confidence intervals (Error ED-1),
none of these differences can be declared significant. They should be
deprioritised in future work.

### 3.4 Global Mean Baseline Is Undefined (Constant Prediction)

The global mean baseline (B1) correctly produces `spearman=None` — a
constant prediction vector has undefined rank correlation. This confirms
the evaluation framework is working correctly and that the task is
non-trivial (the target is not constant).

---

## 4. Most Common Failure Modes (Ranked by Impact)

1. **Era mismatch (DQ-3)** — The most consequential failure. The FBPAC
   dataset is 3–4 years older than the intended analysis period. All
   learned patterns (vocabulary, sponsors, topics) reflect the 2020
   presidential cycle, not the 2022 midterms. Until a 2022–2023 dataset
   is obtained, all model improvements are provisional.

2. **Text explains almost nothing beyond sponsor identity (MC-3/DQ-2)** —
   The gap between the sponsor-mean baseline and the TF-IDF model is
   below 0.01 Spearman. The text model is not meaningfully better than
   simply memorising each advertiser's average impressions.

3. **No confidence intervals (ED-1)** — Without bootstrap CIs or
   repeated runs, the entire experiment matrix is a collection of point
   estimates. Differences smaller than ~0.01 Spearman could easily be
   noise on this val-set size.

4. **Missing temporal features (DQ-1)** — Ad delivery date is NaN for
   all rows. Posting date (election proximity, day of week) is likely
   a strong predictor of impressions that is entirely absent.

5. **Impressions label noise (DQ-4)** — The prediction target is derived
   from the midpoint of coarse ranges (e.g. midpoint of "100K–200K" is
   150K). Ads with very different true reach may have the same label,
   adding irreducible noise to the regression target.

---

## 5. Recommended Priority Order for Next Steps

1. Obtain 2022–2023 individual ad-level data (Ad Library UI exports)
2. Add sponsor as a categorical feature (one-hot or embedding)
3. Sweep vocab to 100k+ to find the saturation point
4. Implement bootstrap CI on val Spearman (n=500 resamples)
5. Add ad length as a feature (character count, word count)
6. Parse and incorporate delivery dates once a new dataset is loaded
