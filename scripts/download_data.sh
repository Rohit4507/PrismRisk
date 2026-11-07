#!/usr/bin/env bash
# Download the Home Credit - Credit Risk Model Stability competition data.
# Requires: a Kaggle account, acceptance of the competition rules (one-time),
# and an API token at ~/.kaggle/kaggle.json (see data/README.md).
# The data is gitignored and may not be redistributed - do NOT commit it.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. One-off install:  pip install kaggle" >&2
  echo "then create a token: Kaggle > Settings > API > Create New Token" >&2
  exit 1
fi

kaggle competitions download -c home-credit-credit-risk-model-stability -p data
unzip -q -o data/home-credit-credit-risk-model-stability.zip -d data
rm data/home-credit-credit-risk-model-stability.zip

echo "Data ready under data/csv_files/ (train/ and test/, ~4 GB)"
echo "Next:  python scripts/prepare_raw.py   # -> data/raw/*.parquet"
