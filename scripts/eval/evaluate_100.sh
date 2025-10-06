#!/usr/bin/env bash
set -euo pipefail

python -m pii_masking.scripts.eval.run_eval \
--hf_dir "/home/mark/Codes/mahdi_codes_folder/axolotl/outputs/pii_masking_mistral/merged_pii_model" \
--gguf   "/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/merged-gguf/mistral7b-pii-Q4_K_M.gguf" \
--samples 100   --split train

