import os, requests, gradio as gr, concurrent.futures

CPU_API = os.getenv("CPU_API_URL", "http://localhost:7860")
GPU_API = os.getenv("GPU_API_URL", "http://localhost:7862")
TITLE = "PII Redactor — CPU vs GPU"

def _call(api_url, text, max_new_tokens):
    r = requests.post(
        f"{api_url}/redact",
        json={"text": text, "max_new_tokens": (max_new_tokens or None)},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("normalized", "")

def redact_both(text, max_new_tokens):
    out_cpu, out_gpu = "", ""
    def _cpu(): return _call(CPU_API, text, max_new_tokens)
    def _gpu(): return _call(GPU_API, text, max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_cpu = ex.submit(_cpu)
        fut_gpu = ex.submit(_gpu)
        for fut, tag in ((fut_cpu, "cpu"), (fut_gpu, "gpu")):
            try:
                val = fut.result()
                if tag == "cpu": out_cpu = val
                else: out_gpu = val
            except Exception as e:
                msg = f"[{tag.upper()} error] {e.__class__.__name__}: {e}"
                if tag == "cpu": out_cpu = msg
                else: out_gpu = msg
    return out_cpu, out_gpu

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    inp = gr.Textbox(label="Input", lines=4, placeholder="Paste text…")
    max_t = gr.Slider(16, 512, value=256, step=16, label="Max new tokens")
    btn = gr.Button("Redact on CPU + GPU")

    with gr.Row():
        out_cpu = gr.Textbox(label="CPU (GGUF)", lines=6)
        out_gpu = gr.Textbox(label="GPU (HF)",  lines=6)

    btn.click(fn=redact_both, inputs=[inp, max_t], outputs=[out_cpu, out_gpu])

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME","0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT","7861"))
    )
