"""
Week 2 baseline:
- Load processed ads data
- Train text-only TF-IDF + Ridge regression on TRAIN
- Evaluate on VALIDATION
- Save stable metric outputs and runtime
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    spearman = spearmanr(y_true, y_pred).correlation
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "spearman": None if pd.isna(spearman) else float(spearman),
        "rmse": rmse,
    }


def run_baseline(
    processed_csv: str | Path,
    output_dir: str | Path = "artifacts/baseline",
    tfidf_max_features: int = 5000,
    ngram_max: int = 2,
    alpha: float = 1.0,
) -> dict:
    t0 = time.perf_counter()

    processed_csv = Path(processed_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_csv)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty. Check data preparation.")

    X_train_text = train_df["text"].fillna("")
    X_val_text = val_df["text"].fillna("")
    y_train = train_df["log_impressions_mid"].to_numpy()
    y_val = val_df["log_impressions_mid"].to_numpy()

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, ngram_max),
        lowercase=False,
        min_df=2,
        max_df=0.95,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    metrics = evaluate_predictions(y_val, val_pred)

    runtime_seconds = time.perf_counter() - t0
    metrics["runtime_seconds"] = round(runtime_seconds, 4)
    metrics["n_train"] = int(len(train_df))
    metrics["n_val"] = int(len(val_df))
    metrics["tfidf_max_features"] = int(tfidf_max_features)
    metrics["model"] = "ridge"
    metrics["target"] = "log_impressions_mid"

    pred_df = val_df[["text", "sponsor", "state", "impressions_mid", "log_impressions_mid"]].copy()
    pred_df["prediction"] = val_pred

    metrics_path = output_dir / "metrics.json"
    preds_path = output_dir / "val_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df.to_csv(preds_path, index=False)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 2 baseline pipeline.")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", default="artifacts/baseline", help="Where to save metrics/predictions")
    parser.add_argument("--tfidf_max_features", type=int, default=5000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    results = run_baseline(
        processed_csv=args.input,
        output_dir=args.output_dir,
        tfidf_max_features=args.tfidf_max_features,
        ngram_max=args.ngram_max,
        alpha=args.alpha,
    )
    print(json.dumps(results, indent=2))