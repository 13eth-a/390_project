"""
Comprehensive experiment sweep across all 4 exploration axes:
  H — Hyperparameters
  P — Prompt / text design
  M — Model variants
  F — Feature engineering

Results written to artifacts/exp_*/metrics.json
Run:  python run_experiments_v2.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler

# ── Load data once ────────────────────────────────────────────────────────────
df    = pd.read_csv("data/processed/political_ads_processed.csv")
train = df[df["split"] == "train"].copy().reset_index(drop=True)
val   = df[df["split"] == "val"].copy().reset_index(drop=True)
y_tr  = train["log_impressions_mid"].to_numpy()
y_val = val["log_impressions_mid"].to_numpy()
X_tr_text  = train["text"].fillna("")
X_val_text = val["text"].fillna("")


# ── Helpers ───────────────────────────────────────────────────────────────────
def sp_rmse(y_true, y_pred):
    sp   = spearmanr(y_true, y_pred).correlation
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return (None if np.isnan(sp) else float(sp)), rmse


def save(name: str, meta: dict):
    p = Path(f"artifacts/{name}")
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {name:35s}  ρ={meta['spearman']}  RMSE={meta['rmse']:.4f}  {meta['runtime_seconds']}s")


def tfidf_ridge(max_features=5000, ngram_max=2, alpha=1.0,
                min_df=2, max_df=0.95, analyzer="word",
                X_tr=None, X_v=None):
    """Fit TF-IDF + Ridge and return (pred, vectorizer, model, seconds)."""
    X_tr  = X_tr  if X_tr  is not None else X_tr_text
    X_v   = X_v   if X_v   is not None else X_val_text
    t0    = time.perf_counter()
    vec   = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max),
                            lowercase=False, min_df=min_df, max_df=max_df,
                            analyzer=analyzer)
    Xtr   = vec.fit_transform(X_tr)
    Xv    = vec.transform(X_v)
    mdl   = Ridge(alpha=alpha, random_state=42)
    mdl.fit(Xtr, y_tr)
    pred  = mdl.predict(Xv)
    return pred, Xtr, Xv, vec, mdl, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS H — Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Axis H: Hyperparameters ──────────────────────────────────────────")

for tag, kw, desc in [
    ("H1_min_df1",   dict(max_features=50000, ngram_max=2, alpha=1.0, min_df=1),
     "50k vocab, min_df=1 (include hapax)"),
    ("H2_min_df5",   dict(max_features=50000, ngram_max=2, alpha=1.0, min_df=5),
     "50k vocab, min_df=5 (prune rare terms)"),
    ("H3_alpha100",  dict(max_features=50000, ngram_max=2, alpha=100.0),
     "50k vocab, Ridge alpha=100 (heavy regularisation)"),
    ("H4_alpha001",  dict(max_features=50000, ngram_max=2, alpha=0.01),
     "50k vocab, Ridge alpha=0.01 (near-OLS)"),
    ("H5_maxdf80",   dict(max_features=50000, ngram_max=2, alpha=1.0, max_df=0.80),
     "50k vocab, max_df=0.80 (tighter stopword removal)"),
]:
    pred, *_, rt = tfidf_ridge(**kw)
    sp, rmse = sp_rmse(y_val, pred)
    save(f"exp_{tag}", {"spearman": sp, "rmse": rmse,
        "runtime_seconds": round(rt, 4), "axis": "H",
        "n_train": len(train), "n_val": len(val), "model": "ridge",
        "tfidf_max_features": kw.get("max_features"),
        "alpha": kw.get("alpha", 1.0), "min_df": kw.get("min_df", 2),
        "max_df": kw.get("max_df", 0.95), "description": desc})


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS P — Prompt / Text design
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Axis P: Prompt / Text design ─────────────────────────────────────")

STOPWORDS = {
    "the","and","for","are","but","not","you","all","can","had","her","was",
    "one","our","out","day","get","has","him","his","how","its","now","old","see",
    "two","way","who","did","did","let","put","say","she","too","use","that","this",
    "with","have","from","they","will","been","were","when","what","which","than",
    "about","just","like","also","some","make","need","there","these","would","could",
    "should","people","every","after","even","most","must","over","such","very","each"
}

def add_stopword_removal(texts):
    """Remove common stopwords from already-lowercased text."""
    def clean(t):
        return " ".join(w for w in str(t).split() if w not in STOPWORDS)
    return texts.apply(clean)

def add_sponsor_prefix(texts, sponsors):
    """Prepend 'SPONSOR_<slug> ' to each ad text."""
    def slug(s):
        s = str(s).lower()
        s = re.sub(r"https?://[^/]+/", "", s)   # strip FB URL prefix
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return f"SPONSOR_{s[:30]}"
    return pd.Series(
        [f"{slug(sp)} {txt}" for sp, txt in zip(sponsors, texts)],
        index=texts.index,
    )

def add_length_bucket(texts):
    """Prepend SHORT / MEDIUM / LONG token based on word count."""
    def bucket(t):
        n = len(str(t).split())
        if n < 20:   tag = "ADLEN_SHORT"
        elif n < 80: tag = "ADLEN_MEDIUM"
        else:         tag = "ADLEN_LONG"
        return f"{tag} {t}"
    return texts.apply(bucket)

# P1: Remove stopwords
X_tr_p1  = add_stopword_removal(X_tr_text)
X_val_p1 = add_stopword_removal(X_val_text)
pred, *_, rt = tfidf_ridge(max_features=50000, ngram_max=2, alpha=1.0,
                            X_tr=X_tr_p1, X_v=X_val_p1)
sp, rmse = sp_rmse(y_val, pred)
save("exp_P1_stopwords_removed", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "P", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "Remove stopwords before TF-IDF (50k vocab)"})

# P2: Sponsor slug prepended to text
X_tr_p2  = add_sponsor_prefix(X_tr_text, train["sponsor"])
X_val_p2 = add_sponsor_prefix(X_val_text, val["sponsor"])
pred, *_, rt = tfidf_ridge(max_features=50000, ngram_max=2, alpha=1.0,
                            X_tr=X_tr_p2, X_v=X_val_p2)
sp, rmse = sp_rmse(y_val, pred)
save("exp_P2_sponsor_prefix", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "P", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "Prepend SPONSOR_<slug> token to ad text (50k vocab)"})

# P3: Length bucket token prepended
X_tr_p3  = add_length_bucket(X_tr_text)
X_val_p3 = add_length_bucket(X_val_text)
pred, *_, rt = tfidf_ridge(max_features=50000, ngram_max=2, alpha=1.0,
                            X_tr=X_tr_p3, X_v=X_val_p3)
sp, rmse = sp_rmse(y_val, pred)
save("exp_P3_length_bucket", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "P", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "Prepend ADLEN_SHORT/MEDIUM/LONG token to ad text (50k vocab)"})

# P4: Character n-grams (3-5) — captures morphology, hashtags, handles
t0 = time.perf_counter()
vec_char = TfidfVectorizer(max_features=50000, analyzer="char_wb",
                           ngram_range=(3, 5), lowercase=False,
                           min_df=2, max_df=0.95)
Xtr_char = vec_char.fit_transform(X_tr_text)
Xv_char  = vec_char.transform(X_val_text)
mdl_char = Ridge(alpha=1.0, random_state=42)
mdl_char.fit(Xtr_char, y_tr)
pred_char = mdl_char.predict(Xv_char)
rt = time.perf_counter() - t0
sp, rmse = sp_rmse(y_val, pred_char)
save("exp_P4_char_ngrams", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "P", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "Character 3-5 grams (50k features) instead of word n-grams"})


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS M — Model variants
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Axis M: Model variants ───────────────────────────────────────────")

# Shared 50k TF-IDF features for all model-variant experiments
t0_vec = time.perf_counter()
_vec50 = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                         lowercase=False, min_df=2, max_df=0.95)
Xtr50 = _vec50.fit_transform(X_tr_text)
Xv50  = _vec50.transform(X_val_text)
vec_time = time.perf_counter() - t0_vec

# M1: Lasso
t0 = time.perf_counter()
mdl = Lasso(alpha=0.001, random_state=42, max_iter=2000)
mdl.fit(Xtr50, y_tr)
pred = mdl.predict(Xv50)
rt = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_M1_lasso", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "M", "model": "lasso",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "Lasso regression (alpha=0.001, 50k TF-IDF)"})

# M2: ElasticNet
t0 = time.perf_counter()
mdl = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=2000)
mdl.fit(Xtr50, y_tr)
pred = mdl.predict(Xv50)
rt = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_M2_elasticnet", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "M", "model": "elasticnet",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "ElasticNet (alpha=0.001, l1_ratio=0.5, 50k TF-IDF)"})

# M3: SVD (200 components) + Ridge — dense low-dim representation
print("  [M3 SVD+Ridge — may take ~30s]")
t0 = time.perf_counter()
svd  = TruncatedSVD(n_components=200, random_state=42)
Xtr_svd = svd.fit_transform(Xtr50)
Xv_svd  = svd.transform(Xv50)
mdl  = Ridge(alpha=1.0, random_state=42)
mdl.fit(Xtr_svd, y_tr)
pred = mdl.predict(Xv_svd)
rt   = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_M3_svd_ridge", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "M", "model": "svd200+ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "TruncatedSVD (200 components) + Ridge on 50k TF-IDF"})

# M4: HistGradientBoosting on SVD-200 features (fast tree-based)
print("  [M4 HistGBT — may take ~30s]")
t0 = time.perf_counter()
mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                    max_leaf_nodes=31, random_state=42)
mdl.fit(Xtr_svd, y_tr)
pred = mdl.predict(Xv_svd)
rt   = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_M4_histgbt", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "M", "model": "histgbt",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "HistGradientBoosting on SVD-200 of 50k TF-IDF"})


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS F — Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Axis F: Feature engineering ──────────────────────────────────────")

def get_text_features(split_df, split_text):
    """Return (n,3) array: [log_char_len, log_word_count, exclamation_density]"""
    chars  = split_text.str.len().fillna(0).to_numpy().astype(float)
    words  = split_text.str.split().str.len().fillna(0).to_numpy().astype(float)
    excl   = split_text.str.count("!").fillna(0).to_numpy().astype(float) / (words + 1)
    quest  = split_text.str.count(r"\?").fillna(0).to_numpy().astype(float) / (words + 1)
    caps   = split_text.str.count(r"[A-Z]").fillna(0).to_numpy().astype(float) / (chars + 1)
    return np.column_stack([
        np.log1p(chars),
        np.log1p(words),
        excl,
        quest,
        caps,
    ])

def get_sponsor_features(split_df, top_sponsors):
    """Binary indicator: is this ad from a top-100 sponsor?"""
    is_top = split_df["sponsor"].isin(top_sponsors).astype(float).to_numpy()
    return is_top.reshape(-1, 1)

top100_sponsors = train["sponsor"].value_counts().head(100).index

F_tr_hand  = get_text_features(train, X_tr_text)
F_val_hand = get_text_features(val,   X_val_text)
S_tr       = get_sponsor_features(train, top100_sponsors)
S_val      = get_sponsor_features(val,   top100_sponsors)

# F1: TF-IDF 50k + 5 hand-crafted text features
t0 = time.perf_counter()
Xtr_f1  = hstack([Xtr50, csr_matrix(F_tr_hand)], format="csr")
Xv_f1   = hstack([Xv50,  csr_matrix(F_val_hand)],  format="csr")
mdl = Ridge(alpha=1.0, random_state=42)
mdl.fit(Xtr_f1, y_tr)
pred = mdl.predict(Xv_f1)
rt = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_F1_text_features", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "F", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "TF-IDF 50k + log_chars, log_words, excl_density, quest_density, caps_ratio"})

# F2: TF-IDF 50k + top-100 sponsor indicator
t0 = time.perf_counter()
Xtr_f2  = hstack([Xtr50, csr_matrix(S_tr)],  format="csr")
Xv_f2   = hstack([Xv50,  csr_matrix(S_val)], format="csr")
mdl = Ridge(alpha=1.0, random_state=42)
mdl.fit(Xtr_f2, y_tr)
pred = mdl.predict(Xv_f2)
rt = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_F2_sponsor_flag", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "F", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "TF-IDF 50k + binary top-100-sponsor indicator"})

# F3: TF-IDF 50k + text features + sponsor flag (kitchen sink)
t0 = time.perf_counter()
Xtr_f3 = hstack([Xtr50, csr_matrix(F_tr_hand), csr_matrix(S_tr)],  format="csr")
Xv_f3  = hstack([Xv50,  csr_matrix(F_val_hand), csr_matrix(S_val)], format="csr")
mdl = Ridge(alpha=1.0, random_state=42)
mdl.fit(Xtr_f3, y_tr)
pred = mdl.predict(Xv_f3)
rt = vec_time + (time.perf_counter() - t0)
sp, rmse = sp_rmse(y_val, pred)
save("exp_F3_all_features", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "F", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 50000,
    "description": "TF-IDF 50k + text stats + sponsor flag (all engineered features)"})

# F4: Text features only (no TF-IDF) — pure structural signal
t0 = time.perf_counter()
all_feats_tr  = np.hstack([F_tr_hand, S_tr])
all_feats_val = np.hstack([F_val_hand, S_val])
mdl = Ridge(alpha=1.0, random_state=42)
mdl.fit(all_feats_tr, y_tr)
pred = mdl.predict(all_feats_val)
rt = time.perf_counter() - t0
sp, rmse = sp_rmse(y_val, pred)
save("exp_F4_no_tfidf", {"spearman": sp, "rmse": rmse,
    "runtime_seconds": round(rt,4), "axis": "F", "model": "ridge",
    "n_train": len(train), "n_val": len(val),
    "tfidf_max_features": 0,
    "description": "Text stats + sponsor flag only (no TF-IDF) — structural baseline"})

print("\nAll axis experiments complete.")
