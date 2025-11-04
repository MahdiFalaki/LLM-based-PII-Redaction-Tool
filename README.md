# 🔒 LLM-Based Text Redaction with Mistral-7B

This project implements an **end-to-end LLM redaction system** optimized for both **GPU** (Transformers) and **CPU** (quantized llama.cpp) environments.
It combines **fine-tuning**, **LoRA merging**, **quantization**, and **Dockerized deployment** via **FastAPI** and **Gradio**, supporting **dual CPU/GPU inference**.

🧠 **Base model:** [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

📚 **Dataset:** [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)

---

## 🌐 Live Demo

> 🚀 Try the quantized model on Hugging Face Spaces:
> [**🔗 https://huggingface.co/spaces/MahdiFalaki/frontend**](https://huggingface.co/spaces/MahdiFalaki/frontend)

⚠️ This demo runs the **quantized GGUF model (CPU)** via `llama.cpp`.
Speed and accuracy are slightly reduced compared to the full-precision GPU version.

---

## 🧩 Overview

| Component | Technology | Description |
|------------|-------------|-------------|
| **Model** | Mistral-7B-Instruct-v0.2 | Base instruction-tuned LLM |
| **Fine-Tuning** | Axolotl + LoRA | PII masking adaptation |
| **Quantization** | llama.cpp (GGUF Q4_K_M / F16) | CPU-optimized inference |
| **Backend** | FastAPI + Pydantic | REST API redaction service |
| **Frontend** | Gradio | Real-time dual-mode web UI |
| **Deployment** | Docker | Modular CPU/GPU containers |

---

## 🧱 Architecture

| Layer | Technology | Purpose |
|:------|:------------|:---------|
| **Model** | Mistral-7B-Instruct | Instruction-tuned backbone |
| **Fine-Tuning** | Axolotl + LoRA | Task-specific adaptation |
| **Quantization** | llama.cpp (GGUF) | Lightweight CPU inference |
| **Backend** | **FastAPI + Pydantic** | REST API for PII redaction |
| **Frontend** | **Gradio** | Interactive web interface |
| **Containerization** | **Docker Compose** | Unified CPU/GPU deployment |

### System Flow

User ─▶ Gradio Frontend
├──▶ FastAPI (GPU: Transformers)
└──▶ FastAPI (CPU: llama.cpp GGUF)
└──▶ Post-processing & Normalization

---

## ⚙️ Local Docker Deployment

Clone the repository inside Axolotl’s `/examples` folder for seamless model training and evaluation.

```bash
# 1️⃣ Clone Axolotl
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl/examples

# 2️⃣ Clone this repo inside /examples
git clone https://github.com/MahdiFalaki/LLM-based-PII-Redaction-Tool.git pii_masking
cd pii_masking
```

---

## 📦 Download Required Model Weights

Before running any containers, ensure the model weights are available locally —
the merged FP16 model (for GPU) and the quantized GGUF model (for CPU).
These are required for **FastAPI** to load the models inside their respective Docker backends.

### 🔹 Option 1 — Fast Deployment (Recommended)

Download the pre-trained weights directly from Hugging Face:

```
pii_masking/
 ├── models/
 │    └── mistral7b-pii-f16.gguf
 └── pii_masking_mistral/
      └── merged_pii_model/
```

### 📥 Download links
- **Quantized CPU Model (GGUF):**
  [quantized_model_cpu.zip](https://huggingface.co/MahdiFalaki/pii-masking-models/resolve/main/quantized_model_cpu.zip)

- **Merged GPU Model (FP16):**
  [merged_pii_model_full.zip](https://huggingface.co/MahdiFalaki/pii-masking-models/resolve/main/merged_pii_model_full.zip)

### 📂 Unzip the merged model
```bash
unzip merged_pii_model_full.zip -d pii_masking_mistral/
```

After unzipping, ensure the structure looks like this:
```
pii_masking/
 ├── models/mistral7b-pii-f16.gguf
 └── pii_masking_mistral/merged_pii_model/config.json
```

Once both models are correctly placed, you can proceed with deployment.

---

### 🔹 Option 2 — Reproduce Weights Yourself

If you have GPU resources and wish to recreate the models:

1️⃣ Fine-tune with **Axolotl (LoRA)**
2️⃣ Merge LoRA weights
3️⃣ Quantize with **llama.cpp**

👉 See the “Training & Quantization Workflow” section later in this README for full reproduction steps.

---

⚠️ **Note:**
- The model weights are large (~11–15 GB each).
- **Do not commit them to GitHub** — keep them local.
- Docker Compose automatically mounts them as volumes during container startup.

---

## 🧩 Start Containers

Once weights are in place:

### CPU Deployment
```
docker compose --profile cpu up --build
```
###  Access
```
Frontend → http://localhost:7861
Backend  → http://localhost:7860
```

###  GPU Deployment
```
docker compose --profile gpu up --build
```
###  Access
```
Frontend → http://localhost:7863
Backend  → http://localhost:7862s
```

### Dual Mode (both backends)
```
docker compose --profile cpu --profile gpu up --build
```

The Gradio interface will show two synchronized outputs:

🧠 GPU (Transformers FP16)
⚙️ CPU (Quantized GGUF)

---

## 🧠 Environment Notes

GPU Backend – runs the merged Transformers model (merged_pii_model) using PyTorch + CUDA.
Requires NVIDIA GPU (≈ 16 GB VRAM recommended).

CPU Backend – loads mistral7b-pii-f16.gguf or Q4_K_M quantized weights through llama-cpp-python.
Requires only CPU threads, no GPU dependencies.

Each backend container has its own Python environment, tailored to its runtime (Torch + CUDA vs llama.cpp bindings).

---

## 💻 Example API Usage

```
curl -X POST http://localhost:7860/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"John Smith lives at 123 Main St, Toronto. CC 4532 9483 0294 5521."}'
```

Response:
```
{
  "normalized": "[FIRSTNAME] [LASTNAME] lives at [ADDRESS]. CC [MASKEDNUMBER].",
  "latency_ms": 1090
}
```
---

## 🔬 Training & Quantization Workflow


```
# 1️⃣ Fine-tune with Axolotl
bash scripts/train/train_full.sh

# 2️⃣ Merge LoRA
python tools/merge.py

# 3️⃣ Convert & Quantize
python llama.cpp/convert_hf_to_gguf.py \
  outputs/pii_masking_mistral/merged_pii_model \
  --outfile models/mistral7b-pii-f16.gguf

llama.cpp/build/bin/quantize \
  models/mistral7b-pii-f16.gguf \
  models/mistral7b-pii-Q4_K_M.gguf Q4_K_M
```

---

## 📈 Performance Summary
| Mode | Engine | Model Type | Speed | Accuracy | Use Case |
|------|---------|-------------|--------|-----------|-----------|
| 🧠 **GPU** | **Transformers** | FP16 Merged | ⚡ Fast | ✅ Highest | Local / Production |
| ⚙️ **CPU** | **llama.cpp** | GGUF F16 / Q4_K_M | 🐢 Slower | ⚠️ Slight drop | Portable / HF Demo |

---

## 🛠 Post-Processing Highlights

- Canonicalizes tag variants (e.g., [Firstname] → [FIRSTNAME])
- Collapses multi-part addresses
- Detects and normalizes credit-card patterns → [MASKEDNUMBER]
- Reduces tag set for lightweight quantized inference

---

## 🧭 Roadmap

- [x] LoRA fine-tuning (Axolotl)
- [x] Model merge & quantization
- [x] FastAPI + Gradio stack
- [x] Dual CPU/GPU Docker deployment
- [x] Hugging Face Space demo (quantized CPU)

---

## 🙏 Credits

Dataset: ai4privacy/pii-masking-200k
Base Model: mistralai/Mistral-7B-Instruct-v0.2
Frameworks: Axolotl · Transformers · FastAPI · Gradio · llama.cpp · Docker

---

## 📜 License

MIT License © 2025 Mahdi Falaki
