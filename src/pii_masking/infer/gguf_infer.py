import os
from llama_cpp import Llama
from pii_masking.utils.prompting import alpaca_prompt

class GGUFModel:
    def __init__(self, gguf_path: str, n_ctx: int = 2048, n_threads: int | None = None):
        self.ll = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads or (os.cpu_count() or 4),
        )

    def generate(self, system: str, user_text: str, max_new_tokens: int = 256) -> str:
        prompt = alpaca_prompt(system=system, instruction="Mask all PII:", input_text=user_text)
        out = self.ll.create_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_new_tokens,
            stop=["</s>"],
        )
        return out["choices"][0]["text"].strip()
