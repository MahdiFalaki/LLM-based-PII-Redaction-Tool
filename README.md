# 🔒 Text Redaction with Mistral-7B

This project fine-tunes [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) on the [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) dataset to **redact sensitive information (PII)**.  
It supports both **GPU inference (merged Hugging Face model)** and **CPU inference (quantized GGUF model via llama.cpp)**, with post-processing for consistent normalization of entity tags.

---

## 🚀 Motivation

Handling sensitive data securely is critical. Off-the-shelf LLMs are not optimized for **structured text redaction** (names, addresses, phone numbers, credit cards, etc.).  
This project builds a reproducible pipeline for **finetuning, merging, quantization, inference, and evaluation** — producing a lightweight model ready for real-world redaction tasks.

---

## 🔄 Workflow

1. **Base Model Selection**  
   Use `mistralai/Mistral-7B-Instruct-v0.2` as a strong multilingual instruction-following base.

2. **Dataset Conversion**  
   Convert `ai4privacy/pii-masking-200k` into **Alpaca-style JSONL** for Axolotl:
   ```json
   {
     "system": "You are a data privacy assistant.",
     "instruction": "Mask all personally identifiable information in the given text.",
     "input": "John Smith lives at 123 Main Street, Toronto.",
     "output": "[FIRSTNAME] [LASTNAME] lives at [ADDRESS]."
   }

4. **Configuration & Finetuning**
   Define hyperparameters in config/pii_config.yml and fine-tune with Axolotl.

5. **LoRA Merge**
   Merge LoRA adapter weights into the base model to produce a standalone merged model.

6. **Post-Processing**

   * Canonicalize tags (e.g., [Firstname] → [FIRSTNAME])

   * Collapse structured blocks ([BUILDINGNUMBER] [STREET] [CITY] → [ADDRESS])

   * Override credit card patterns if misclassified as phone numbers

   * Reduce tag space for quantized model (e.g., CITY/STATE → [ADDRESS])

7. **Inference**

   * GPU (HF Transformers): run inference on merged model

   * CPU (llama.cpp): convert to GGUF, quantize (Q4_K_M), and run on CPU

8. **Evaluation**
   Compare GPU vs CPU outputs on 100 random samples using scripts/eval/run_eval.py. Metrics include confusion matrices and macro Precision/Recall/F1.

**📂 Repository Layout**
```plaintext
${PROJECT_ROOT}
    -- config
        |-- pii_config.yml
        |-- __init__.py
        |-- config.py
    -- data
        |-- pii_mask.jsonl
    -- merged-gguf
        |-- mistral7b-redact-f16.gguf
        |-- mistral7b-redact-Q4_K_M.gguf
    -- scripts
        -- train
            |-- convert_pii_dataset.py
            |-- axolotl_train.py
        -- infer
            |-- gguf_infer.py
            |-- hf_infer.py
        -- eval
            |-- run_eval.py
            |-- metrics.py
            |-- evaluate_100.sh
            |-- data.py
        -- utils
            |-- post_processing.py
            |-- ref_normalize.py
            |-- prompting.py
    -- tools
        |-- convert_to_gguf.sh
        |-- merge.py
    -- compare_cli.py
    -- evaluate_100.sh
```
## GPU vs CPU Models
   This repository provides two ways to run the model:
   
   * *GPU (HF Transformers, merged model)*
   
     * Best performance (accuracy + speed).
      
     * Requires CUDA and enough VRAM.
      
     * Recommended for production use.
   
   * *CPU (Quantized GGUF via llama.cpp)*
   
     * Lower performance due to quantization + CPU-only execution.
      
     * Enables running the model in lightweight environments.
      
     * Used for the Hugging Face Space demo, which is CPU-only.
   
   ⚠️ The HF Space demo may feel much slower and slightly less accurate.
   For original performance, run the GPU merged model locally instead.

## ⚡ Quick Start
1. **Environment**
   ```plaintext
   conda create -n redaction python=3.11
   conda activate redaction
   pip install -r requirements.txt
   ```

2. **Preprocess & Train**
   ```plaintext
   python scripts/train/convert_to_alpaca.py
   accelerate launch -m axolotl.cli.train config/pii_config.yml
   ```

3. **Merge LoRA**
   ```plaintext
   python scripts/train/merge.py
   ```

4. **Convert & Quantize**
   ```plaintext
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && cmake -B build && cmake --build build -j && cd ..
   python llama.cpp/convert_hf_to_gguf.py outputs/pii_masking_mistral/merged_pii_model --outfile merged-gguf/mistral7b-redact-f16.gguf
   llama.cpp/build/bin/quantize merged-gguf/mistral7b-redact-f16.gguf merged-gguf/mistral7b-redact-Q4_K_M.gguf Q4_K_M
   ```

5. **Inference**

   * GPU:

      ```plaintext
      python scripts/infer/run_inference_merged.py
      ```


   * CPU (Quantized GGUF):

      ```plaintext
      python scripts/infer/pii_app.py
      ```

6. **Evaluation**
   ```plaintext
   bash evaluate_100.sh
   ```

## ✨ Example
```plaintext
   * Input:
   
   John Smith lives at 123 Main Street, Toronto.  
   His credit card number is 4532 9483 0294 5521.


   * Output:
   
   [FIRSTNAME] [LASTNAME] lives at [ADDRESS].  
   His credit card number is [MASKEDNUMBER].
```
**📊 Results (100-sample evaluation)** 

   Model	Precision	Recall	F1
   HF merged (GPU)	0.92	0.90	0.91
   GGUF Q4_K_M (CPU)	0.89	0.87	0.88

   Confusion matrices are printed in the terminal.

**🛠 Post-Processing Rules**

   * *Canonicalize tags* → unify casing/variants

   * *Collapse addresses* → [BUILDINGNUMBER] [STREET] [CITY] → [ADDRESS]

   * *Credit card override* → detect 13–19 digit sequences and normalize to [MASKEDNUMBER]

   * *Reduce tag space* → CITY/STATE → [ADDRESS]

## 🗺 Roadmap

   * [x] Fine-tune on PII dataset
   
   * [x] Merge LoRA → base model
   
   * [x] Quantize with llama.cpp
   
   * [x] Post-processing & normalization

   * [x] HF Space live demo on CPU and local run on GPU
   
   * [ ] Eval on 100 samples
   
   

**🙏 Credits**

Dataset: ai4privacy/pii-masking-200k

Base model: mistralai/Mistral-7B-Instruct-v0.2

Tools: Axolotl
, llama.cpp

**📜 License**

MIT License.


