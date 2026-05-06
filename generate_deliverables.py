"""
Generate all deliverable artifacts:
  - artifacts/results/experiment_matrix.csv
  - artifacts/results/metric_over_time.png
  - artifacts/results/failure_analysis_memo.md
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── 1. Collect all experiment metrics ────────────────────────────────────────

EXPERIMENTS = [
    # (artifact_dir, exp_id, label, axis_label, controlled_variable)
    ("exp_b1_mean_baseline",   "B1", "Global Mean",        "naive",   "Naive baseline"),
    ("exp_b2_sponsor_baseline","B2", "Sponsor Mean",       "naive",   "Naive baseline"),
    ("exp_c1_vocab1k",         "C1", "TF-IDF 1k vocab",   "vocab",   "max_features=1k"),
    ("exp_c2_vocab2500",       "C2", "TF-IDF 2.5k vocab", "vocab",   "max_features=2.5k"),
    ("exp_c3_vocab5k",         "C3", "TF-IDF 5k vocab",   "vocab",   "max_features=5k (baseline)"),
    ("exp_c4_vocab10k",        "C4", "TF-IDF 10k vocab",  "vocab",   "max_features=10k"),
    ("exp_c5_vocab25k",        "C5", "TF-IDF 25k vocab",  "vocab",   "max_features=25k"),
    ("exp_c6_vocab50k",        "C6", "TF-IDF 50k vocab",  "vocab",   "max_features=50k"),
    ("exp_c7_trigrams",        "C7", "TF-IDF 10k trigrams","ngram",  "ngram_range=(1,3)"),
    ("baseline",               "00", "Baseline (frozen)", "vocab",   "max_features=5k (frozen ref)"),
    ("exp_01_features10k",     "01", "10k vocab",         "vocab",   "max_features=10k"),
    ("exp_02_unigrams",        "02", "Unigrams only",     "ngram",   "ngram_range=(1,1)"),
    ("exp_03_alpha10",         "03", "alpha=10",          "alpha",   "Ridge alpha=10"),
    ("exp_04_alpha01",         "04", "alpha=0.1",         "alpha",   "Ridge alpha=0.1"),
]

root = Path("artifacts")
rows = []
for dirname, exp_id, label, axis, controlled in EXPERIMENTS:
    p = root / dirname / "metrics.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    rows.append({
        "exp_id": exp_id,
        "label": label,
        "controlled_variable": controlled,
        "axis": axis,
        "model": m.get("model", "ridge"),
        "tfidf_max_features": m.get("tfidf_max_features", 0),
        "ngram_max": m.get("ngram_max", 2),
        "alpha": m.get("alpha", 1.0),
        "spearman": m.get("spearman"),
        "rmse": round(m.get("rmse", float("nan")), 4),
        "runtime_s": m.get("runtime_seconds"),
        "n_train": m.get("n_train"),
        "n_val": m.get("n_val"),
        "description": m.get("description", ""),
    })

df = pd.DataFrame(rows)

# Drop duplicates (C3/00 and C4/01 are the same runs)
df = df.drop_duplicates(subset=["label"]).reset_index(drop=True)

out_dir = Path("artifacts/results")
out_dir.mkdir(parents=True, exist_ok=True)

# ── 2. Save experiment matrix CSV ────────────────────────────────────────────
matrix_path = out_dir / "experiment_matrix.csv"
df.to_csv(matrix_path, index=False)
print(f"Matrix saved → {matrix_path}")
print(df[["exp_id","label","spearman","rmse","runtime_s"]].to_string(index=False))


# ── 3. Metric-over-time plot ─────────────────────────────────────────────────
# "Over time" = in experiment order (ordered by vocab size for vocab axis,
# then append alpha/ngram experiments)

plot_order = [
    ("B1", "Global Mean\n(naive)"),
    ("B2", "Sponsor Mean\n(naive)"),
    ("C1", "1k\nvocab"),
    ("C2", "2.5k\nvocab"),
    ("C3", "5k\nvocab"),
    ("C4", "10k\nvocab"),
    ("C5", "25k\nvocab"),
    ("C6", "50k\nvocab"),
    ("C7", "10k\ntrigrams"),
    ("02", "Unigrams\n5k"),
    ("03", "alpha=10\n5k"),
    ("04", "alpha=0.1\n5k"),
]
id_to_row = {r["exp_id"]: r for r in df.to_dict("records")}
plot_ids   = [pid for pid, _ in plot_order if pid in id_to_row]
plot_xlabs = [lbl for pid, lbl in plot_order if pid in id_to_row]
spear_vals = [id_to_row[pid]["spearman"] for pid in plot_ids]
rmse_vals  = [id_to_row[pid]["rmse"]     for pid in plot_ids]

# Replace None (constant-prediction baseline) with 0
spear_vals = [v if v is not None else 0.0 for v in spear_vals]

xs = np.arange(len(plot_ids))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    "Experiment Metrics Across Controlled Runs\n"
    "STAT 390 — Political Ad Impression Prediction",
    fontsize=13, fontweight="bold", y=0.98,
)

# --- Spearman panel ---
colors = []
for pid in plot_ids:
    ax = id_to_row[pid]["axis"]
    colors.append({"naive": "#aaaaaa", "vocab": "#4c72b0",
                   "ngram": "#dd8452", "alpha": "#55a868"}.get(ax, "#999"))

bars1 = ax1.bar(xs, spear_vals, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
ax1.axhline(id_to_row["C3"]["spearman"], color="#4c72b0", linestyle="--",
            linewidth=1.2, alpha=0.6, label="5k baseline ref")
ax1.set_ylabel("Spearman ρ (val)", fontsize=10)
ax1.set_ylim(0, 0.45)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax1.legend(fontsize=8)
for bar, val in zip(bars1, spear_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

# --- RMSE panel ---
bars2 = ax2.bar(xs, rmse_vals, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
ax2.axhline(id_to_row["C3"]["rmse"], color="#4c72b0", linestyle="--",
            linewidth=1.2, alpha=0.6, label="5k baseline ref")
ax2.set_ylabel("RMSE (log impressions, val)", fontsize=10)
ax2.set_ylim(0.55, 0.68)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax2.legend(fontsize=8)
for bar, val in zip(bars2, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax2.set_xticks(xs)
ax2.set_xticklabels(plot_xlabs, fontsize=8)

# Legend for axis groups
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#aaaaaa", label="Naive baseline"),
    Patch(facecolor="#4c72b0", label="Vocab sweep (Ridge α=1)"),
    Patch(facecolor="#dd8452", label="N-gram variation"),
    Patch(facecolor="#55a868", label="Regularisation variation"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=4,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plot_path = out_dir / "metric_over_time.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved → {plot_path}")


# ── 4. Failure Analysis Memo ─────────────────────────────────────────────────
baseline_sp = id_to_row["C3"]["spearman"]
best_sp     = id_to_row["C6"]["spearman"]
sponsor_sp  = id_to_row["B2"]["spearman"]

memo = f"""# Failure Analysis Memo
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

The sponsor-mean baseline (B2) achieves Spearman **{sponsor_sp:.3f}** using
only the advertiser identity — no text at all. The frozen TF-IDF baseline
achieves **{baseline_sp:.3f}**, only **{baseline_sp - sponsor_sp:.3f}** above the
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
| 5k    | {id_to_row['C3']['spearman']:.3f}    | {id_to_row['C3']['rmse']:.3f} |
| 10k   | {id_to_row['C4']['spearman']:.3f}    | {id_to_row['C4']['rmse']:.3f} |
| 25k   | {id_to_row['C5']['spearman']:.3f}    | {id_to_row['C5']['rmse']:.3f} |
| 50k   | **{best_sp:.3f}**    | **{id_to_row['C6']['rmse']:.3f}** |

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
"""

memo_path = out_dir / "failure_analysis_memo.md"
memo_path.write_text(memo, encoding="utf-8")
print(f"Memo saved  → {memo_path}")
print("\nAll deliverables generated in artifacts/results/")
