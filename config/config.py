# src/pii_masking/config/config.py
import os

DEFAULT_HF_DIR = os.getenv(
    "PII_DEFAULT_HF_DIR",
    "/examples/pii_masking/pii_masking_mistral/merged_pii_model",
)
DEFAULT_GGUF = os.getenv(
    "PII_DEFAULT_GGUF",
    "/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/merged-gguf/mistral7b-pii-Q4_K_M.gguf",
)

BASE_MODEL = os.getenv("PII_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

SYSTEM_PROMPT = os.getenv("PII_SYSTEM_PROMPT", "You are a data privacy assistant.")
N_CTX = int(os.getenv("PII_N_CTX", "2048"))
CPU_THREADS = None if os.getenv("PII_CPU_THREADS", "") == "" else int(os.getenv("PII_CPU_THREADS"))
