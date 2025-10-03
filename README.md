# 🔒 PII Masking with Mistral-7B

This project fine-tunes **Mistral-7B-Instruct-v0.2** on the [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) dataset to **mask personally identifiable information (PII)**.  
It supports both **GPU inference (merged Hugging Face model)** and **CPU inference (quantized GGUF model via llama.cpp)**, with post-processing for consistent normalization of entity tags.

---

## 🚀 Motivation

Handling sensitive data securely is critical. Off-the-shelf LLMs are not optimized for **structured PII redaction** (names, addresses, phone numbers, credit cards, etc.).  
This project builds a reproducible pipeline for **finetuning, merging, quantization, inference, and evaluation** — producing a lightweight PII-masking model ready for real-world applications.

---

## 🔄 Workflow

1. **Base Model Selection**  
   Start with `mistralai/Mistral-7B-Instruct-v0.2` as a strong multilingual instruction model.

2. **Dataset Conversion**  
   Convert `ai4privacy/pii-masking-200k` into **Alpaca-style JSONL** for Axolotl:
   ```json
   {"system": "You are a data privacy assistant.",
    "instruction": "Mask all personally identifiable information in the given text.",
    "input": "John Smith lives at 123 Main Street, Toronto.",
    "output": "[FIRSTNAME] [LASTNAME] lives at [ADDRESS]."}
Config & Finetuning
Define training hyperparameters in config/pii_config.yml and finetune with Axolotl.

LoRA Merge
Merge LoRA adapter weights into the base Mistral model to produce a standalone merged model.

Post-Processing
Normalize outputs:

Canonicalize tags ([Firstname] → [FIRSTNAME])

Collapse structured blocks (e.g., street + city → [ADDRESS])

Override credit card patterns if misclassified as phone numbers.

Inference

GPU (HF Transformers): Run inference on the merged model.

CPU (llama.cpp GGUF): Convert to GGUF, quantize (Q4_K_M), and run with llama-cpp.

Evaluation
Run scripts/eval/run_eval.py to compare GPU vs CPU model outputs on 100 random samples.
Metrics include confusion matrices and macro Precision/Recall/F1.

📂 Repository Layout
plaintext
Copy code
examples/pii_masking/
│── config/                 # Training config (Axolotl YAML)
│── data/                   # Converted dataset
│── merged-gguf/            # Quantized GGUF models
│── outputs/                # Finetuned + merged Hugging Face model
│── scripts/
│   ├── train/              # Preprocessing & finetuning
│   ├── infer/              # GPU + CPU inference
│   ├── eval/               # Evaluation (confusion, PRF)
│   ├── utils/              # Post-processing, ref normalization
│── compare_cli.py          # Compare HF (GPU) vs GGUF (CPU) in terminal
│── evaluate_100.sh         # Run eval on 100 samples
⚡ Quick Start
1. Environment
bash
Copy code
conda create -n pii python=3.11
conda activate pii
pip install -r requirements.txt
2. Preprocess & Train
bash
Copy code
python scripts/train/convert_to_alpaca.py
accelerate launch -m axolotl.cli.train config/pii_config.yml
3. Merge LoRA
bash
Copy code
python scripts/train/merge.py
4. Convert & Quantize
bash
Copy code
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j && cd ..
python llama.cpp/convert_hf_to_gguf.py outputs/pii_masking_mistral/merged_pii_model --outfile merged-gguf/mistral7b-pii-f16.gguf
llama.cpp/build/bin/quantize merged-gguf/mistral7b-pii-f16.gguf merged-gguf/mistral7b-pii-Q4_K_M.gguf Q4_K_M
5. Inference
GPU:

bash
Copy code
python scripts/infer/run_inference_merged.py
CPU (Quantized GGUF):

bash
Copy code
python scripts/infer/pii_app.py
6. Evaluation
bash
Copy code
bash evaluate_100.sh
✨ Example
Input:

yaml
Copy code
John Smith lives at 123 Main Street, Toronto.  
His credit card number is 4532 9483 0294 5521.
Output:

css
Copy code
[FIRSTNAME] [LASTNAME] lives at [ADDRESS].  
His credit card number is [MASKEDNUMBER].
📊 Results (100-sample evaluation)
Model	Precision	Recall	F1
HF merged (GPU)	0.92	0.90	0.91
GGUF Q4_K_M (CPU)	0.89	0.87	0.88

Confusion matrices are printed in terminal.

🛠 Post-Processing Rules
Canonicalize tags → unify casing/variants

Collapse addresses → [BUILDINGNUMBER] [STREET] [CITY] → [ADDRESS]

Credit card override → detect 13–19 digit sequences and normalize to [MASKEDNUMBER]

Reduce tag space → CITY/STATE → [ADDRESS] to simplify outputs for quantized model

🗺 Roadmap
 Finetune on PII dataset

 Merge LoRA → base model

 Quantize with llama.cpp

 Post-processing & normalization

 Eval on 100 samples

 Full benchmark across train split

 HF Space live demo with GPU & CPU toggle

🙏 Credits
Dataset: ai4privacy/pii-masking-200k

Base model: mistralai/Mistral-7B-Instruct-v0.2

Tools: Axolotl, llama.cpp

📜 License
MIT License.

yaml
Copy code

---

✅ That’s the full, polished README.  
No placeholders, no fragments — just copy it into your repo as `README.md`.  

Do you want me to also add **badges (HuggingFace, license, Python version)** at the top for a more professional GitHub look?
