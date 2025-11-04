import os, gradio as gr
from llama_cpp import Llama
from examples.pii_masking.src.pii_masking.utils.post_processing import normalize_entities

GGUF = "/home/mark/Codes/mahdi_codes_folder/axolotl/examples/pii_masking/merged-gguf/mistral7b-pii-Q4_K_M.gguf"

ll = Llama(
    model_path=GGUF,
    n_ctx=1024,
    n_threads=os.cpu_count() or 4,
    n_batch=256,
)

def mistral_prompt(system, user):
    return f"<s>[INST] {system}\n\n{user} [/INST]"

SYSTEM = "You are a data privacy assistant."

def redact(text: str):
    prompt = mistral_prompt(SYSTEM, f"Mask all PII: {text}")
    out = ll.create_completion(
        prompt=prompt,
        temperature=0.0,
        max_tokens=256,
        stop=["</s>"],
    )
    raw = out["choices"][0]["text"]
    normalized = normalize_entities(raw, system=SYSTEM, user_text=text)
    return raw, normalized

demo = gr.Interface(
    fn=redact,
    inputs=gr.Textbox(lines=4, label="Enter text"),
    outputs=[gr.Textbox(label="Raw model output"),
             gr.Textbox(label="Normalized output")],
    title="PII Redaction (CPU demo, quantized)",
    description="Mistral-7B-Instruct fine-tuned, quantized to Q4_K_M via llama.cpp. CPU-only demo."
)

if __name__ == "__main__":
    demo.launch()
