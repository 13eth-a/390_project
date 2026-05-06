"""Run all controlled experiments and save metrics JSON for each."""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def spearman_rmse(y_true, y_pred):
    sp = spearmanr(y_true, y_pred).correlation
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return (None if np.isnan(sp) else float(sp)), rmse


df = pd.read_csv("data/processed/political_ads_processed.csv")
train = df[df["split"] == "train"].copy()
val   = df[df["split"] == "val"].copy()
y_train = train["log_impressions_mid"].to_numpy()
y_val   = val["log_impressions_mid"].to_numpy()
X_train_text = train["text"].fillna("")
X_val_text   = val["text"].fillna("")

def save(name, metrics):
    p = Path(f"artifacts/{name}")
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"{name}: spearman={metrics['spearman']}, rmse={metrics['rmse']:.4f}, t={metrics['runtime_seconds']}s")


# ── B1: Global mean baseline ────────────────────────────────────────────────
t0 = time.perf_counter()
pred = np.full(len(y_val), y_train.mean())
sp, rmse = spearman_rmse(y_val, pred)
save("exp_b1_mean_baseline", {
    "spearman": sp, "rmse": rmse,
    "runtime_seconds": round(time.perf_counter()-t0, 4),
    "n_train": len(train), "n_val": len(val),
    "model": "mean_baseline", "tfidf_max_features": 0,
    "description": "Predict global train mean for every ad"
})

# ── B2: Sponsor mean baseline ────────────────────────────────────────────────
t0 = time.perf_counter()
sponsor_mean = train.groupby("sponsor")["log_impressions_mid"].mean()
global_mean  = y_train.mean()
pred2 = val["sponsor"].map(sponsor_mean).fillna(global_mean).to_numpy()
sp, rmse = spearman_rmse(y_val, pred2)
save("exp_b2_sponsor_baseline", {
    "spearman": sp, "rmse": rmse,
    "runtime_seconds": round(time.perf_counter()-t0, 4),
    "n_train": len(train), "n_val": len(val),
    "model": "sponsor_mean_baseline", "tfidf_max_features": 0,
    "description": "Predict per-sponsor mean impressions from train set"
})

# ── Helper: TF-IDF + Ridge ───────────────────────────────────────────────────
def run_tfidf_ridge(max_features, ngram_max, alpha, name, description):
    t0 = time.perf_counter()
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max),
                          lowercase=False, min_df=2, max_df=0.95)
    X_tr = vec.fit_transform(X_train_text)
    X_v  = vec.transform(X_val_text)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_tr, y_train)
    pred = model.predict(X_v)
    sp, rmse = spearman_rmse(y_val, pred)
    save(name, {
        "spearman": sp, "rmse": rmse,
        "runtime_seconds": round(time.perf_counter()-t0, 4),
        "n_train": len(train), "n_val": len(val),
        "model": "ridge", "tfidf_max_features": max_features,
        "ngram_max": ngram_max, "alpha": alpha,
        "description": description
    })

# ── Vocab sweep (single axis: max_features) ──────────────────────────────────
run_tfidf_ridge(1000,  2, 1.0, "exp_c1_vocab1k",   "Vocab sweep: 1k features")
run_tfidf_ridge(2500,  2, 1.0, "exp_c2_vocab2500",  "Vocab sweep: 2.5k features")
run_tfidf_ridge(5000,  2, 1.0, "exp_c3_vocab5k",    "Vocab sweep: 5k features (= baseline)")
run_tfidf_ridge(10000, 2, 1.0, "exp_c4_vocab10k",   "Vocab sweep: 10k features")
run_tfidf_ridge(25000, 2, 1.0, "exp_c5_vocab25k",   "Vocab sweep: 25k features")
run_tfidf_ridge(50000, 2, 1.0, "exp_c6_vocab50k",   "Vocab sweep: 50k features")

# ── Trigrams ─────────────────────────────────────────────────────────────────
run_tfidf_ridge(10000, 3, 1.0, "exp_c7_trigrams",   "10k features, trigrams (1-3)")

print("\nAll experiments done.")
