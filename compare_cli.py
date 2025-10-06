# examples/pii_masking/compare_cli.py
import os, sys, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

# local utils
from pii_masking.scripts.utils.prompting import mistral_inst
from pii_masking.scripts.utils.post_processing import normalize_entities
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a data privacy assistant."

def load_hf(hf_dir: str):
    tok = AutoTokenizer.from_pretrained(hf_dir, use_fast=True)

    # Force full model to GPU
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},   # put all modules on cuda:0
        low_cpu_mem_usage=True,
    )
    return tok, model, "cuda"


def gen_hf(tok, model, device, src, max_new_tokens=128, temperature=0.0):
    import torch
    from scripts.utils.post_processing import normalize_entities

    msgs = [
        {"role": "system", "content": "You are a data privacy assistant."},
        {"role": "user", "content": f"Mask all PII: {src}"},
    ]

    # Build prompt and ensure tensors + attention_mask are on the same device as the model
    input_ids = tok.apply_chat_template(
        msgs,
        return_tensors="pt",
        add_generation_prompt=True,   # helps ensure we generate after the user turn
    ).to(device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # Greedy decoding (temperature ignored by design here)
    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,     # <-- fixes the attention_mask warning
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

    decoded = tok.decode(out_ids[0], skip_special_tokens=True)

    # Only return the normalized text (no raw), and without the "Mask all PII:" prefix
    normalized = normalize_entities(
        decoded,
        system="You are a data privacy assistant.",
        user_text=src,
    )
    return normalized

def load_gguf(gguf_path, n_ctx=1024, n_threads=None):
    if n_threads is None:
        n_threads = max(2, (os.cpu_count() or 4) - 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("LLAMA_NUM_THREADS", str(n_threads))
    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    return llm

def gen_gguf(llm, src, max_new_tokens=256):
    from scripts.utils.post_processing import normalize_entities

    prompt = f"<s>[INST] You are a data privacy assistant.\n\nMask all PII: {src} [/INST]"
    out = llm.create_completion(prompt=prompt, max_tokens=max_new_tokens, stop=["</s>"])
    raw = out["choices"][0]["text"]
    normalized = normalize_entities(raw, system="You are a data privacy assistant.", user_text=src)
    return normalized

def main():
    ap = argparse.ArgumentParser(description="Compare GPU (HF) vs CPU (GGUF) PII masking in terminal.")
    ap.add_argument("--hf_dir", required=True, help="Path to merged HF model dir")
    ap.add_argument("--gguf",   required=True, help="Path to GGUF model file")
    ap.add_argument("--text",   help="Single input string")
    ap.add_argument("--infile", help="Path to text file (one input per line)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n_ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for llama.cpp")
    ap.add_argument("--jsonl_out", help="Optional path to save results as JSONL")
    args = ap.parse_args()

    if not args.text and not args.infile:
        ap.error("Provide either --text or --infile")

    # load models once
    print(f"[HF] loading from {args.hf_dir} …")
    tok, hf_model, device = load_hf(args.hf_dir)
    print(f"[GGUF] loading from {args.gguf} …")
    llm = load_gguf(args.gguf, n_ctx=args.n_ctx, n_threads=args.threads)

    # gather inputs
    inputs = []
    if args.text:
        inputs.append(args.text.strip())
    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            inputs.extend([ln.strip() for ln in f if ln.strip()])

    # optional log
    f_log = open(args.jsonl_out, "w", encoding="utf-8") if args.jsonl_out else None

    for i, src in enumerate(inputs, 1):
        print("\n" + "="*80)
        print(f"[{i}] INPUT:")
        print(src)

        # after you compute both
        hf_norm = gen_hf(tok, hf_model, device, src, args.max_new_tokens, args.temperature)
        print("\n--- HF (GPU) NORMALIZED ---")
        print(hf_norm)

        gg_norm  = gen_gguf(llm, src, args.max_new_tokens)
        print("\n--- GGUF (CPU) NORMALIZED ---")
        print(gg_norm)
        print("="*80)

        if f_log:
            f_log.write(json.dumps({
                "input": src,
                "hf_norm": hf_norm,
                "gguf_norm": gg_norm
            }, ensure_ascii=False) + "\n")

    if f_log:
        f_log.close()
        print(f"\nSaved JSONL to: {args.jsonl_out}")

if __name__ == "__main__":
    # ensure local package import works when run from repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)  # makes `pii_masking` importable
    main()
