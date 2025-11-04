#!/usr/bin/env bash
set -euo pipefail

HF_DIR="${HF_DIR:-/home/mark/Codes/mahdi_codes_folder/axolotl/outputs/pii_masking_mistral/merged_pii_model}"
GGUF="${GGUF:-/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/merged-gguf/mistral7b-pii-Q4_K_M.gguf}"
TEXT="${TEXT:-John Smith lives at 123 Main Street, Toronto. CC 4532 9483 0294 5521.}"
INFILE="${INFILE:-}"
JSONL_OUT="${JSONL_OUT:-compare_log.jsonl}"

if [[ -n "$INFILE" ]]; then
  python -m pii_masking.cli.compare --hf_dir "$HF_DIR" --gguf "$GGUF" --infile "$INFILE" --jsonl_out "$JSONL_OUT"
else
  python -m pii_masking.cli.compare --hf_dir "$HF_DIR" --gguf "$GGUF" --text "$TEXT" --jsonl_out "$JSONL_OUT"
fi
