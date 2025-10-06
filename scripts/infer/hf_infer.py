# examples/pii_masking/models/hf_infer.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pii_masking.scripts.prompting import mistral_messages

class HFModel:
    def __init__(self, model_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = (
            torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if self.device == "cuda" else torch.float32)
        )
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

    def generate(self, system: str, user_text: str, max_new_tokens=256):
        msgs = mistral_messages(system, f"Mask all PII: {user_text}")
        enc = self.tok.apply_chat_template(msgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                enc, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False
            )
        # keep special tokens for safer downstream stripping
        return self.tok.decode(out[0], skip_special_tokens=False)
