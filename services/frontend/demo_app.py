import os
import time

from fastapi import FastAPI
import gradio as gr
import requests
import uvicorn

CPU_API = os.getenv("CPU_API_URL", "http://127.0.0.1:7860")
TITLE = "PII Redaction Demo"
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("PII_MAX_NEW_TOKENS", "128"))
REQUEST_TIMEOUT = int(os.getenv("PII_REQUEST_TIMEOUT", "120"))


def _api_get(path: str) -> dict:
    response = requests.get(f"{CPU_API}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def load_model_info():
    try:
        payload = _api_get("/")
    except Exception as e:
        return (
            "Backend unavailable",
            (
                "### Model Info\n"
                f"- Backend: `{CPU_API}`\n"
                f"- Status: `Unavailable ({e.__class__.__name__})`\n"
                f"- Max new tokens: `{DEFAULT_MAX_NEW_TOKENS}`"
            ),
        )

    return (
        "Backend connected",
        (
            "### Model Info\n"
            f"- Backend: `{CPU_API}`\n"
            f"- Model: `{payload.get('model_name', 'unknown')}`\n"
            f"- GGUF path: `{payload.get('gguf', 'unknown')}`\n"
            f"- Context window: `{payload.get('n_ctx', 'unknown')}`\n"
            f"- Threads: `{payload.get('threads', 'unknown')}`\n"
            f"- Max new tokens: `{DEFAULT_MAX_NEW_TOKENS}`"
        ),
    )


def redact_text(user_text: str):
    user_text = (user_text or "").strip()
    if not user_text:
        return "", "### Status\n- Enter text to redact.", "### Model Info\n- Press Refresh Status to load backend info."

    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"{CPU_API}/redact",
            json={"text": user_text, "max_new_tokens": DEFAULT_MAX_NEW_TOKENS},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        status, model_info = load_model_info()
        return (
            "",
            (
                "### Status\n"
                f"- Request status: `Failed ({e.__class__.__name__})`\n"
                f"- Backend status: `{status}`"
            ),
            model_info,
        )

    round_trip_ms = (time.perf_counter() - t0) * 1000.0
    status_md = (
        "### Status\n"
        f"- Request status: `OK`\n"
        f"- Model latency: `{(payload.get('latency_ms') or 0.0):.1f} ms`\n"
        f"- Round-trip latency: `{round_trip_ms:.1f} ms`\n"
        f"- Detected tags: `{payload.get('tag_count') or 0}`\n"
        f"- Model: `{payload.get('model_name') or 'unknown'}`"
    )
    model_info = (
        "### Model Info\n"
        f"- Backend: `{CPU_API}`\n"
        f"- Model: `{payload.get('model_name') or 'unknown'}`\n"
        f"- GGUF path: `{payload.get('model_path') or 'unknown'}`\n"
        f"- Max new tokens: `{payload.get('max_new_tokens') or DEFAULT_MAX_NEW_TOKENS}`"
    )
    return payload.get("normalized", ""), status_md, model_info


def clear_demo():
    status, model_info = load_model_info()
    return "", "", f"### Status\n- {status}", model_info


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(
        f"# {TITLE}\n"
        "Single-model CV demo for PII redaction using the local GGUF deployment path."
    )

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Input Text",
                lines=8,
                placeholder="Enter text containing names, addresses, emails, phone numbers, or account details.",
            )
            with gr.Row():
                run_btn = gr.Button("Redact Text", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            status_md = gr.Markdown("### Status\n- Loading backend status...")
            model_info_md = gr.Markdown("### Model Info\n- Loading model info...")

    output_text = gr.Textbox(label="Redacted Output", lines=8)

    gr.Examples(
        examples=[
            ["John Smith lives at 123 Main St and uses john@example.com for banking updates."],
            ["Hey my name is Sarah, I study in Toronto and my phone number is 416-555-1234."],
        ],
        inputs=[input_text],
    )

    run_btn.click(
        fn=redact_text,
        inputs=[input_text],
        outputs=[output_text, status_md, model_info_md],
        show_api=False,
    )
    clear_btn.click(
        fn=clear_demo,
        inputs=[],
        outputs=[input_text, output_text, status_md, model_info_md],
        show_api=False,
    )
    demo.load(
        fn=load_model_info,
        inputs=[],
        outputs=[status_md, model_info_md],
        show_api=False,
    )

app = gr.mount_gradio_app(FastAPI(), demo, path="/")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
    )
