#!/usr/bin/env bash
set -euo pipefail

HF_DIR="${HF_DIR:-/home/mark/Codes/mahdi_codes_folder/axolotl/outputs/pii_masking_mistral/merged_pii_model}"
GGUF="${GGUF:-/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/merged-gguf/mistral7b-pii-Q4_K_M.gguf}"
JSONL="${JSONL:-/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/data/synthetic100_eval.jsonl}"
OUTDIR="${OUTDIR:-eval_out}"

python -m pii_masking.eval.run_eval \
  --hf_dir "$HF_DIR" \
  --gguf   "$GGUF" \
  --jsonl  "$JSONL" \
  --plot \
  --outdir "$OUTDIR"
