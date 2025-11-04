import os
from llama_cpp import Llama
from pii_masking.utils.prompting import mistral_inst

class GGUFModel:
    def __init__(self, gguf_path: str, n_ctx: int = 2048, n_threads: int | None = None):
        self.ll = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads or (os.cpu_count() or 4),
        )

    def generate(self, system: str, user_text: str, max_new_tokens: int = 256) -> str:
        prompt = mistral_inst(system, f"Mask all PII: {user_text}").lstrip("<s>")
        out = self.ll.create_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_new_tokens,
            stop=["</s>"],
        )
        return out["choices"][0]["text"].strip()
