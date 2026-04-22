from src.data_loader import load_and_prepare_ads
from src.baseline_pipeline import run_baseline

RAW_PATH = "data/raw/fbpac-ads-en-US.csv"
PROCESSED_PATH = "data/processed/political_ads_processed.csv"

load_and_prepare_ads(
    csv_path=RAW_PATH,
    min_year=2022,
    max_year=2023,
    save_processed_path=PROCESSED_PATH,
)

results = run_baseline(
    processed_csv=PROCESSED_PATH,
    output_dir="artifacts/baseline",
)

print(results)