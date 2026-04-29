# Research Log — Modeling Political Ad Messaging Effectiveness

---

## Week 3 — Dry-Run Experiments

**Date:** April 29, 2026  
**Goal:** Sanity-check the frozen baseline pipeline and probe sensitivity to basic hyperparameters before any systematic search in Week 3.  
**Dataset:** `data/processed/political_ads_processed.csv` (113,698 train / 24,296 val)  
**Evaluation:** Spearman rank correlation and RMSE on the validation set.

All runs use the same processed CSV and deterministic splits. The test set was not touched.

---

### Experiment 00 — Baseline (reference)

**Command:**
```bash
python run_baseline.py
```

**Config:**

| Parameter | Value |
|-----------|-------|
| `tfidf_max_features` | 5,000 |
| `ngram_range` | (1, 2) |
| Ridge `alpha` | 1.0 |

**Results:**

| Metric | Value |
|--------|-------|
| Spearman (val) | **0.3333** |
| RMSE (val) | 0.6111 |
| Runtime | 13.6 s |

**Notes:** Pipeline ran cleanly end-to-end. This run defines the frozen baseline against which all future experiments are compared.

---

### Experiment 01 — Larger Vocabulary (10k features)

**Command:**
```bash
python -m src.baseline_pipeline \
  --input data/processed/political_ads_processed.csv \
  --output_dir artifacts/exp_01_features10k \
  --tfidf_max_features 10000 --ngram_max 2 --alpha 1.0
```

**Change from baseline:** `tfidf_max_features` doubled from 5,000 → 10,000.

**Results:**

| Metric | Value | Δ vs baseline |
|--------|-------|---------------|
| Spearman (val) | **0.3505** | **+0.0172** |
| RMSE (val) | 0.6045 | **−0.0066** |
| Runtime | 14.6 s | +1.0 s |

**Notes:** Best result of all dry runs. Doubling the vocabulary gives a measurable improvement in both ranking quality and absolute error at negligible runtime cost. Suggests the 5k vocab in the baseline is too restrictive for this corpus size (>100k ads). Will revisit in Week 3 with a wider vocabulary sweep.

---

### Experiment 02 — Unigrams Only

**Command:**
```bash
python -m src.baseline_pipeline \
  --input data/processed/political_ads_processed.csv \
  --output_dir artifacts/exp_02_unigrams \
  --tfidf_max_features 5000 --ngram_max 1 --alpha 1.0
```

**Change from baseline:** `ngram_range` restricted to (1, 1) — bigrams removed.

**Results:**

| Metric | Value | Δ vs baseline |
|--------|-------|---------------|
| Spearman (val) | 0.3325 | −0.0008 |
| RMSE (val) | 0.6109 | −0.0002 |
| Runtime | 6.6 s | **−7.0 s** |

**Notes:** Performance is essentially identical to the baseline (difference well within noise). Bigrams are contributing almost nothing when the vocabulary is capped at 5k, because slots are occupied by the same high-frequency unigrams. Benefit of bigrams only likely appears with a larger vocabulary. Unigrams-only is 2× faster — useful as a cheap approximation during rapid iteration.

---

### Experiment 03 — High Regularization (alpha = 10)

**Command:**
```bash
python -m src.baseline_pipeline \
  --input data/processed/political_ads_processed.csv \
  --output_dir artifacts/exp_03_alpha10 \
  --tfidf_max_features 5000 --ngram_max 2 --alpha 10.0
```

**Change from baseline:** Ridge `alpha` increased 10× from 1.0 → 10.0.

**Results:**

| Metric | Value | Δ vs baseline |
|--------|-------|---------------|
| Spearman (val) | 0.3370 | +0.0037 |
| RMSE (val) | 0.6118 | +0.0007 |
| Runtime | 14.9 s | +1.3 s |

**Notes:** Slightly better ranking (Spearman +0.004) but slightly worse RMSE. Consistent with the model being mildly under-regularized at alpha=1.0 for this feature space. The effect is small; alpha is not a high-value tuning target on its own.

---

### Experiment 04 — Low Regularization (alpha = 0.1)

**Command:**
```bash
python -m src.baseline_pipeline \
  --input data/processed/political_ads_processed.csv \
  --output_dir artifacts/exp_04_alpha01 \
  --tfidf_max_features 5000 --ngram_max 2 --alpha 0.1
```

**Change from baseline:** Ridge `alpha` decreased 10× from 1.0 → 0.1.

**Results:**

| Metric | Value | Δ vs baseline |
|--------|-------|---------------|
| Spearman (val) | 0.3312 | −0.0021 |
| RMSE (val) | 0.6120 | +0.0009 |
| Runtime | 23.2 s | **+9.6 s** |

**Notes:** Both metrics slightly worse than baseline. Notably, runtime jumped to 23s despite having the same feature set — Ridge's iterative solver requires more iterations to converge under low regularization. No benefit to reducing alpha here.

---

## Summary Table — All Week 2 Dry Runs

| Exp | max_features | ngram_max | alpha | Spearman | RMSE | Runtime (s) |
|-----|-------------|-----------|-------|----------|------|-------------|
| 00 Baseline | 5,000 | 2 | 1.0 | 0.3333 | 0.6111 | 13.6 |
| 01 10k vocab | **10,000** | 2 | 1.0 | **0.3505** | **0.6045** | 14.6 |
| 02 Unigrams | 5,000 | **1** | 1.0 | 0.3325 | 0.6109 | **6.6** |
| 03 High α | 5,000 | 2 | **10.0** | 0.3370 | 0.6118 | 14.9 |
| 04 Low α | 5,000 | 2 | **0.1** | 0.3312 | 0.6120 | 23.2 |

**Best Spearman:** Exp 01 (0.3505)  
**Best RMSE:** Exp 01 (0.6045)  
**Fastest run:** Exp 02 (6.6s — good for rapid iteration)

---

## Reflection on Agent Performance

### What the Agent Did Well

**1. Code generation and execution was reliable.**  
The agent correctly invoked `baseline_pipeline.py` via the module CLI interface (`python -m src.baseline_pipeline`) rather than editing source files. All experiments produced parseable JSON output without requiring any manual fixes.

**2. Artifact organization.**  
The agent routed each experiment's outputs to a separate `artifacts/exp_XX_*/` directory, keeping results isolated and avoiding accidental overwrites of the frozen baseline in `artifacts/baseline/`.

**3. Respected the frozen evaluation contract.**  
The agent never touched the test split, never changed the metric schema, and never altered the train/val assignment logic. The frozen contract defined in the README was honored throughout.

**4. Parallel experiment design was logical.**  
The agent identified three independent axes of variation (vocabulary size, n-gram range, regularization strength) and tested each in isolation — consistent with a controlled experiment mindset.

**5. Summarization was accurate.**  
The results table was assembled correctly from raw JSON outputs, and the direction of each delta was correctly interpreted (e.g., lower RMSE is better).

### What the Agent Did Badly

**1. No statistical uncertainty estimates.**  
All experiments ran exactly once. The differences between configurations (e.g., Spearman 0.333 vs 0.332) are well within the noise floor for a single val-set evaluation, yet the agent treated them as reliable point estimates. Proper practice would run each config 3+ times with different random seeds or use bootstrap CI on the val set.

**2. Hyperparameter range was narrow and uninformed.**  
The agent tested only three alpha values (0.1, 1, 10) and two vocab sizes (5k, 10k) — all round numbers. A principled search would use a log-spaced grid or Bayesian optimization. The "best" config found (10k vocab) was the simplest possible increment, not the result of a real search.

**3. No exploration of the prediction residuals.**  
The agent collected Spearman and RMSE but never examined *which* ads were predicted poorly. Systematic error analysis (e.g., are long ads harder to predict? ads from small sponsors?) is essential for guiding feature engineering but was not done.

**4. Feature space is still very shallow.**  
Sponsor identity, posting date, state targeting, and ad length are all available in the processed CSV but were never incorporated. The agent stayed only in the TF-IDF parameter space, which limits what can be learned from these experiments.

**5. No discussion of what the Spearman value means substantively.**  
Spearman ≈ 0.33 is weak. The agent never reflected on whether this is meaningfully useful or just barely better than a random ranking, which is important for framing the project's ambition.

---

## Common Failure Modes Observed

| # | Failure Mode | Description | Frequency |
|---|-------------|-------------|-----------|
| 1 | **Metric conflation** | Treating small Spearman differences (< 0.005) as meaningful signal when there is no confidence interval. Results from Exp 02–04 appear to vary, but differences are likely noise. | Every run |
| 2 | **Vocabulary cap defeats n-gram benefit** | At 5k features with bigrams enabled, the vocab slots are dominated by high-frequency unigrams. Bigrams offer no benefit at this cap (Exp 02 ≈ Exp 00). This is a known TF-IDF pitfall that was not anticipated in the baseline design. | Exps 00/02 |
| 3 | **Slow convergence under low regularization** | Ridge with `alpha=0.1` took 23s vs 14s for `alpha=1.0` on the same feature set. The solver iterates longer without strong regularization, which is a hidden cost in automated searches. | Exp 04 |
| 4 | **No baseline for "useful" performance** | There is no majority-class baseline, mean-prediction baseline, or sponsor-frequency baseline to compare against. Without this, Spearman=0.333 is hard to contextualize. | All exps |
| 5 | **Single-point evaluation** | Each experiment was run once. Stochastic elements (even if minor with Ridge + fixed seed) and OS-level timing variation mean runtime and metric comparisons between experiments can be misleading without averaging across runs. | All exps |

---

## Next Steps (Week 3)

- [ ] Add a constant-prediction baseline (predict global train mean for every ad)
- [ ] Add a sponsor-frequency baseline (predict mean impressions per sponsor)
- [ ] Run top config (10k vocab, bigrams) with 3 seeds to estimate metric variance
- [ ] Sweep `max_features` more finely: [5k, 7.5k, 10k, 20k, 50k]
- [ ] Add sponsor and ad-length features alongside TF-IDF
- [ ] Implement bootstrap CI on val Spearman
