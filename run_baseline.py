import argparse
from src.data_loader import load_and_prepare_ads
from src.baseline_pipeline import run_baseline

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw/fbpac-ads-en-US.csv",
                    help="Path to raw ads CSV (default: FBPAC dataset)")
parser.add_argument("--processed", default="data/processed/political_ads_processed.csv",
                    help="Path to write/read processed CSV")
parser.add_argument("--output-dir", default="artifacts/baseline",
                    help="Where to save metrics and predictions")
parser.add_argument("--min-year", type=int, default=2022)
parser.add_argument("--max-year", type=int, default=2023)
args = parser.parse_args()

load_and_prepare_ads(
    csv_path=args.raw,
    min_year=args.min_year,
    max_year=args.max_year,
    save_processed_path=args.processed,
)

results = run_baseline(
    processed_csv=args.processed,
    output_dir=args.output_dir,
)

print(results)