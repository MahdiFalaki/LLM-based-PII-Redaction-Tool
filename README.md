# PII Redaction Platform

Production-ready PII masking pipeline built on Mistral-7B, with:

- Fine-tuning workflow (Axolotl + LoRA)
- CPU and GPU inference backends
- FastAPI backend
- Gradio web app for side-by-side inference
- End-to-end Docker deployment

Live demo: https://huggingface.co/spaces/MahdiFalaki/frontend

## Supported Labels

`[NAME] [ADDRESS] [CARDNUMBER] [PHONENUMBER] [DATE] [EMAIL] [URL] [USERNAME] [IP] [IPV4] [IPV6] [ACCOUNTNUMBER] [OTHERPII]`

## Project Specs

- Dataset (converted): `209,261` samples
- Dataset (English-only training set): `43,501` samples (`43,491` after sequence-length filtering)
- LoRA config: `r=32`, `alpha=16`, `dropout=0.05`
- Trainable parameters: `83,886,080` (`1.1451%` of `7,325,618,176`)
- Training device: `NVIDIA GeForce RTX 3090 (24 GB VRAM)`
- Driver/CUDA used: `Driver 580.126.09`, `CUDA 13.0`

## System Architecture

- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Training: Axolotl + LoRA
- CPU Inference: llama.cpp / GGUF
- GPU Inference: Hugging Face Transformers
- API: FastAPI
- UI: Gradio
- Orchestration: Docker Compose

## Quick Start (Local Deployment)

### 1. Clone

```bash
git clone https://github.com/MahdiFalaki/LLM-based-PII-Redaction-Tool.git
cd LLM-based-PII-Redaction-Tool
```

### 2. Download model artifacts

Place artifacts in this structure:

```text
.
└── outputs/
    ├── gguf/
    │   └── pii_masking_english_basic_v1/
    │       └── quantized/
    │           └── mistral7b-pii-Q5_K_M.gguf
    └── pii_masking_mistral_english_basic_v1/
        └── merged_pii_model/
            ├── config.json
            ├── tokenizer_config.json
            └── ...
```

Downloads:

- CPU GGUF: https://huggingface.co/MahdiFalaki/pii-masking-models/resolve/main/quantized_model_cpu.zip
- GPU merged model: https://huggingface.co/MahdiFalaki/pii-masking-models/resolve/main/merged_pii_model_full.zip

Example:

```bash
mkdir -p outputs/gguf/pii_masking_english_basic_v1/quantized \
  outputs/pii_masking_mistral_english_basic_v1
# unzip downloads here
```

### 3. Start services

CPU profile:

```bash
docker compose --profile cpu up --build
```

GPU profile:

```bash
docker compose --profile gpu up --build
```

Dual profile (recommended for side-by-side comparison):

```bash
docker compose --profile cpu --profile gpu up --build
```

Endpoints:

- Frontend: `http://localhost:7861`
- CPU API: `http://localhost:7860`
- GPU API: `http://localhost:7862`

Notes:

- Open the UI at `http://localhost:7861`, not `0.0.0.0:7861`
- The frontend expects the merged HF model and quantized GGUF artifacts to already exist
- If you update frontend code, rebuild that image before restarting:

```bash
docker compose build frontend
```

## Demo Lite

For a lightweight single-model demo that reuses the existing local GGUF artifact without copying it:

```bash
docker compose -f docker-compose.demo.yml up --build
```

Demo endpoints:

- Demo UI: `http://localhost:7861`
- Demo API: `http://localhost:7860`

The demo path uses:

- one GGUF model
- one CPU backend
- one simplified Gradio UI
- no arena, leaderboard, or model switching

## API Usage

```bash
curl -X POST http://localhost:7860/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"John Smith lives at 123 Main St. Card 4532 9483 0294 5521."}'
```

## Training

### 1. Install training dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt
```

Install merge and evaluation dependencies as needed:

```bash
pip install -r requirements-tools.txt
pip install -r requirements-eval.txt
```

If your environment uses an Axolotl source checkout, export:

```bash
export AXOLOTL_SRC=/path/to/axolotl/src
```

### 2. Run training (English + basic profile)

```bash
bash src/pii_masking/train/train_full.sh
```

### 3. Training configuration defaults

Canonical contract:

- [config/experiment_contract.json](/home/mark/Projects/pii_masking/config/experiment_contract.json)
- English-only
- `basic` tag profile
- frozen `train` / `validation` / `test` splits
- generated dataset artifacts live under `data/contracts/` and are not source-controlled

Training config defaults in `config/pii_config.yml`.

Training data roles:

- `train.jsonl`: optimization / gradient updates
- `validation.jsonl`: in-training evaluation and best-checkpoint selection
- `test.jsonl`: final held-out evaluation only

## Merge and Quantize

After LoRA training:

```bash
python tools/merge.py \
  --root outputs/pii_masking_mistral_english_basic_v1 \
  --out outputs/pii_masking_mistral_english_basic_v1/merged_pii_model \
  --base_model mistralai/Mistral-7B-Instruct-v0.2
```

Build GGUF f16 + quantization matrix with llama.cpp:

```bash
bash tools/quantize_matrix.sh
# outputs in:
#   outputs/gguf/pii_masking_english_basic_v1/f16/
#   outputs/gguf/pii_masking_english_basic_v1/quantized/
#   outputs/gguf/pii_masking_english_basic_v1/manifest.json
```

`tools/quantize_matrix.sh` now derives its default `MERGED` and `OUTDIR` paths from the canonical experiment contract, so the default repeatable flow is:

```bash
python tools/merge.py
bash tools/quantize_matrix.sh
```

## Evaluation and Model Testing

Canonical benchmark on the frozen test split:

```bash
python -m pii_masking.eval.benchmark_quant \
  --hf_dir outputs/pii_masking_mistral_english_basic_v1/merged_pii_model \
  --gguf_dir outputs/gguf/pii_masking_english_basic_v1/quantized \
  --glob "*.gguf" \
  --split test \
  --samples 500 \
  --outdir src/pii_masking/eval/eval_runs
```

Evaluation notes:

- canonical evaluation reads the frozen contract split artifacts instead of re-splitting the raw Hugging Face dataset
- `--samples 500` is a faster benchmark run
- `--samples 0` evaluates the full frozen test split
- leaderboard and summary outputs in `src/pii_masking/eval/eval_runs` include contract metadata so results stay tied to the dataset definition

## Product Roadmap

1. Evaluation improvements
   - Expand test suites by domain and edge-case categories
   - Add automated regression scoring per label class
2. Model testing
   - Build benchmark packs for precision/recall and hallucination checks
   - Track model versions with reproducible test reports
3. Backend and frontend improvements
   - Throughput optimization and concurrency profiling
   - UX iteration for side-by-side output review workflows
4. Cloud deployment
   - Deploy managed online endpoints on AWS
   - Add production-grade observability and autoscaling
5. Bedrock comparison
   - Run side-by-side online A/B evaluation with AWS Bedrock models
   - Publish comparative latency, quality, and cost reports

## Repository Layout

```text
config/                     # training/runtime config and experiment contract
data/contracts/             # generated frozen dataset artifacts 
services/
  backend/                  # FastAPI CPU/GPU APIs
  frontend/                 # Gradio app
src/pii_masking/
  train/                    # training pipeline
  infer/                    # inference wrappers
  eval/                     # evaluation scripts
  utils/                    # post-processing, metrics, prompt helpers
tools/                      # merge/quantization helpers
```

## Credits

- Dataset: `ai4privacy/pii-masking-200k`
- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Stack: Axolotl, Transformers, FastAPI, Gradio, llama.cpp, Docker

## License

MIT
