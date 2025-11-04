import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pii_masking.utils.prompting import mistral_inst

class HFModel:
    def __init__(self, model_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        # ensure pad token id
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        ).eval()

        if self.device == "cpu":
            self.model = self.model.to("cpu")

    def generate(self, system: str, user_text: str, max_new_tokens: int = 256) -> str:
        # Use the exact same prompt as GGUF
        prompt = mistral_inst(system, f"Mask all PII: {user_text}")

        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.tok.pad_token_id).long()
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        gen_ids = out[0, input_ids.shape[1]:]
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()

