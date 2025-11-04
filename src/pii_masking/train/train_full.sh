#!/usr/bin/env bash
# ==========================================================
# TRAINING PIPELINE
# ==========================================================

set -Eeuo pipefail
IFS=$'\n\t'
: "${DRY_RUN:=0}"

run() {
  printf '+'
  for arg in "$@"; do printf ' %q' "$arg"; done
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$PROJECT_ROOT/config/pii_config.yml" ]]:
  CONFIG_FILE="$PROJECT_ROOT/config/pii_config.yml"
else
  echo "❌ Missing pii_config.yml in configs/ or config/"; exit 1
fi

DATA_OUT="$PROJECT_ROOT/data/pii_mask.jsonl"

echo "🔍 Checking environment..."
command -v python >/dev/null || { echo "❌ Python not found"; exit 1; }
command -v accelerate >/dev/null || echo "⚠️ accelerate not found (required for training)"

echo "📘 Step 1: Convert dataset to Alpaca format..."
if [[ -f "$DATA_OUT" ]]; then
  echo "ℹ️  Dataset already exists at $DATA_OUT (skipping conversion)"
else
  run python -m pii_masking.train.convert_dataset
fi

echo "🚀 Step 2: Launch training with Axolotl..."
# axolotl_train resolves the config path by itself
run python -m pii_masking.train.axolotl_train

echo "✅ Training pipeline completed successfully!"
