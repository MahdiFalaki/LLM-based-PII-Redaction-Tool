#!/usr/bin/env bash
set -euo pipefail

MERGED="${MERGED:-/home/mark/Codes/mahdi_codes_folder/axolotl/outputs/pii_masking_mistral/merged_pii_model}"
OUTDIR="${OUTDIR:-./merged-gguf}"
F16_OUT="${F16_OUT:-$OUTDIR/mistral7b-pii-f16.gguf}"
Q4_OUT="${Q4_OUT:-$OUTDIR/mistral7b-pii-Q4_K_M.gguf}"

LLAMA_DIR="${LLAMA_DIR:-./third_party/llama.cpp}"
BUILD_DIR="$LLAMA_DIR/build"
LLAMA_GGUF_BIN="$BUILD_DIR/bin/llama-gguf"
LLAMA_QUANT_BIN="$BUILD_DIR/bin/llama-quantize"

echo "=== Ensure llama.cpp ==="
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

echo "=== Build (no CURL) ==="
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DLLAMA_CURL=OFF >/dev/null
cmake --build "$BUILD_DIR" -j >/dev/null

mkdir -p "$OUTDIR"

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

python - <<'PY' >/dev/null 2>&1 || pip install --quiet sentencepiece protobuf
try:
    import sentencepiece, google.protobuf  # noqa
except Exception:
    raise SystemExit(1)
PY

echo "=== Convert HF -> GGUF (fp16) ==="
if [ -n "$CONVERTER_PY" ]; then
  echo "Using converter: $CONVERTER_PY"
  set +e
  if [[ "$CONVERTER_PY" == *"convert.py" ]]; then
    python "$CONVERTER_PY" hf-to-gguf --model "$MERGED" --outfile "$F16_OUT" --outtype f16 || \
    python "$CONVERTER_PY" hf-to-gguf --model "$MERGED" --output  "$F16_OUT" --outtype f16
  else
    python "$CONVERTER_PY" "$MERGED" --outfile "$F16_OUT" --outtype f16 || \
    python "$CONVERTER_PY" "$MERGED" --output  "$F16_OUT" --outtype f16
  fi
  RET=$?
  set -e
  if [ $RET -ne 0 ]; then
    echo "❌ Python converter failed."; exit $RET
  fi
else
  echo "⚠️ Python converter not found. Trying binary."
  if [ ! -x "$LLAMA_GGUF_BIN" ]; then
    echo "❌ No converter found."; exit 1
  fi
  "$LLAMA_GGUF_BIN" --input "$MERGED" --output "$F16_OUT" --outtype f16
fi
echo "✓ $F16_OUT"

echo "=== Quantize to Q4_K_M ==="
if [ ! -x "$LLAMA_QUANT_BIN" ]; then
  echo "❌ quantize not found at $LLAMA_QUANT_BIN"; exit 1
fi
"$LLAMA_QUANT_BIN" "$F16_OUT" "$Q4_OUT" Q4_K_M
echo "✓ $Q4_OUT"
