# src/pii_masking/cli/compare.py
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

from pii_masking.utils.post_processing import normalize_entities
from pii_masking.utils.prompting import mistral_messages, mistral_inst

SYSTEM = "You are a data privacy assistant."

def load_hf(hf_dir: str):
    tok = AutoTokenizer.from_pretrained(hf_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model = model.to("cpu")
    return tok, model, device

def gen_hf(tok, model, device, src, max_new_tokens=128):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Mask all PII: {src}"}]
    input_ids = tok.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    ).to(device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(out_ids[0], skip_special_tokens=False)
    return normalize_entities(decoded, system=SYSTEM, user_text=src)

def load_gguf(gguf_path, n_ctx=2048, n_threads=None):
    if n_threads is None:
        n_threads = max(2, (os.cpu_count() or 4) - 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("LLAMA_NUM_THREADS", str(n_threads))
    return Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

def gen_gguf(llm, src, max_new_tokens=256):
    prompt = mistral_inst(SYSTEM, f"Mask all PII: {src}")
    out = llm.create_completion(prompt=prompt, max_tokens=max_new_tokens, stop=["</s>"])
    raw = out["choices"][0]["text"]
    return normalize_entities(raw, system=SYSTEM, user_text=src)

def main():
    ap = argparse.ArgumentParser(description="Compare GPU (HF) vs CPU (GGUF) PII masking in terminal.")
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--text")
    ap.add_argument("--infile")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n_ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--jsonl_out")
    args = ap.parse_args()

    if not args.text and not args.infile:
        ap.error("Provide either --text or --infile")

    tok, hf_model, device = load_hf(args.hf_dir)
    llm = load_gguf(args.gguf, n_ctx=args.n_ctx, n_threads=args.threads)

    inputs = []
    if args.text:
        inputs.append(args.text.strip())
    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            inputs.extend([ln.strip() for ln in f if ln.strip()])

    f_log = open(args.jsonl_out, "w", encoding="utf-8") if args.jsonl_out else None

    for i, src in enumerate(inputs, 1):
        print("\n" + "="*80)
        print(f"[{i}] INPUT:\n{src}")

        hf_norm = gen_hf(tok, hf_model, device, src, args.max_new_tokens)
        print("\n--- HF (GPU) NORMALIZED ---")
        print(hf_norm)

        gg_norm = gen_gguf(llm, src, args.max_new_tokens)
        print("\n--- GGUF (CPU) NORMALIZED ---")
        print(gg_norm)
        print("="*80)

        if f_log:
            f_log.write(json.dumps({"input": src, "hf_norm": hf_norm, "gguf_norm": gg_norm}, ensure_ascii=False) + "\n")

    if f_log:
        f_log.close()
        print(f"\nSaved JSONL to: {args.jsonl_out}")

if __name__ == "__main__":
    main()
