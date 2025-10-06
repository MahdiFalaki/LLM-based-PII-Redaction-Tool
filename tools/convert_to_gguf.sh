#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# Paths (assumes you run from examples/pii_masking)
# ---------------------------------------
# Edit or export MERGED to point to your merged HF model folder if needed:
#   export MERGED="/home/mark/Codes/mahdi_codes_folder/axolotl/outputs/pii_masking_mistral/merged_pii_model"
MERGED="/home/mark/Codes/mahdi_codes_folder/axolotl/outputs/pii_masking_mistral/merged_pii_model"
OUTDIR="/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/merged-gguf"
F16_OUT="$OUTDIR/mistral7b-pii-f16.gguf"
Q4_OUT="$OUTDIR/mistral7b-pii-Q4_K_M.gguf"

LLAMA_DIR="./llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
LLAMA_GGUF_BIN="$BUILD_DIR/bin/llama-gguf"
LLAMA_QUANT_BIN="$BUILD_DIR/bin/llama-quantize"

echo "=== Ensure llama.cpp exists ==="
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

echo "=== Configure + build (no CURL) ==="
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD_DIR" -j >/dev/null
echo "✓ Built llama.cpp tools"

mkdir -p "$OUTDIR"

# ---------------------------------------
# Prefer the Python converter with underscores
# ---------------------------------------
CONVERTER_PY=""
for p in \
  "$LLAMA_DIR/convert_hf_to_gguf.py" \
  "$LLAMA_DIR/convert_hf_to_gguf_update.py" \
  "$LLAMA_DIR/converters/convert.py" \
  "$LLAMA_DIR/convert.py" \
  "$LLAMA_DIR/scripts/convert-hf-to-gguf.py" \
  "$LLAMA_DIR/convert-hf-to-gguf.py"
do
  if [ -f "$p" ]; then CONVERTER_PY="$p"; break; fi
done

# Minimal deps
python - <<'PY' >/dev/null 2>&1 || pip install --quiet sentencepiece protobuf
try:
    import sentencepiece, google.protobuf
except Exception:
    raise SystemExit(1)
PY

echo "=== Convert HF -> GGUF (fp16) ==="
if [ -n "$CONVERTER_PY" ]; then
  echo "Using converter: $CONVERTER_PY"
  set +e
  # Try --outfile first (most variants), then --output
  if [[ "$CONVERTER_PY" == *"convert.py" ]]; then
    # unified converter: subcommand style
    python "$CONVERTER_PY" hf-to-gguf --model "$MERGED" --outfile "$F16_OUT" --outtype f16
    RET=$?
    if [ $RET -ne 0 ]; then
      python "$CONVERTER_PY" hf-to-gguf --model "$MERGED" --output  "$F16_OUT" --outtype f16
      RET=$?
    fi
  else
    # dedicated scripts
    python "$CONVERTER_PY" "$MERGED" --outfile "$F16_OUT" --outtype f16
    RET=$?
    if [ $RET -ne 0 ]; then
      python "$CONVERTER_PY" "$MERGED" --output  "$F16_OUT" --outtype f16
      RET=$?
    fi
  fi
  set -e
  if [ $RET -ne 0 ]; then
    echo "❌ Python converter failed. Check the error above."
    exit $RET
  fi
else
  echo "⚠️  Python converter not found. Falling back to binary."
  if [ ! -x "$LLAMA_GGUF_BIN" ]; then
    echo "❌ No converter found (python or binary). Update llama.cpp."
    exit 1
  fi
  "$LLAMA_GGUF_BIN" --input "$MERGED" --output "$F16_OUT" --outtype f16
fi
echo "✓ Wrote: $F16_OUT"

echo "=== Quantize to Q4_K_M (CPU friendly) ==="
if [ ! -x "$LLAMA_QUANT_BIN" ]; then
  echo "❌ quantize binary not found at $LLAMA_QUANT_BIN"
  exit 1
fi
"$LLAMA_QUANT_BIN" "$F16_OUT" "$Q4_OUT" Q4_K_M
echo "✓ Wrote: $Q4_OUT"

echo "✅ All done."
