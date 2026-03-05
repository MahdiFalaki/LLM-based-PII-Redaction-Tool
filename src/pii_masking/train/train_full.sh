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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
if [[ -n "${AXOLOTL_SRC:-}" ]]; then
  export PYTHONPATH="${AXOLOTL_SRC}:${PYTHONPATH}"
  echo "ℹ️  Using AXOLOTL_SRC=${AXOLOTL_SRC}"
fi

if [[ -f "$PROJECT_ROOT/config/pii_config.yml" ]]; then
  CONFIG_FILE="$PROJECT_ROOT/config/pii_config.yml"
else
  echo "❌ Missing pii_config.yml in configs/ or config/"; exit 1
fi

DATA_OUT="$PROJECT_ROOT/data/pii_mask.jsonl"

echo "🔍 Checking environment..."
command -v python >/dev/null || { echo "❌ Python not found"; exit 1; }
command -v accelerate >/dev/null || { echo "❌ accelerate not found (required for training)"; exit 1; }

run python - <<'PY'
import importlib.util
required = {
    "datasets": "datasets",
    "axolotl.cli.preprocess": "axolotl",
}
missing = [name for mod, name in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    raise SystemExit("Missing required Python modules: " + ", ".join(sorted(set(missing))))
print("✅ Python package check passed: datasets, axolotl")
PY

echo "📘 Step 1: Convert dataset to Alpaca format..."
if [[ -n "${PII_LANG:-}" ]]; then
  echo "ℹ️  Applying language filter: PII_LANG=${PII_LANG}"
fi
echo "ℹ️  Tag profile: PII_TAG_PROFILE=${PII_TAG_PROFILE:-full}"
if [[ -f "$DATA_OUT" && "${FORCE_REBUILD_DATA:-0}" != "1" ]]; then
  echo "ℹ️  Dataset already exists at $DATA_OUT (skipping conversion)"
else
  run python -m pii_masking.train.convert_dataset
fi

run python - <<'PY'
from pathlib import Path
from hashlib import sha256
p = Path("data/pii_mask.jsonl")
if not p.exists():
    raise SystemExit(f"Missing dataset file: {p}")
with p.open("rb") as f:
    digest = sha256(f.read()).hexdigest()
rows = sum(1 for _ in p.open("r", encoding="utf-8"))
print(f"✅ Dataset ready: {p} rows={rows} sha256={digest[:16]}")
PY

echo "🚀 Step 2: Launch training with Axolotl..."
# axolotl_train resolves the config path by itself
run python -m pii_masking.train.axolotl_train

echo "✅ Training pipeline completed successfully!"
