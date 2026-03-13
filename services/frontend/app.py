import concurrent.futures
import json
import os
import time
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse, Response
import gradio as gr
import requests
import uvicorn

CPU_API = os.getenv("CPU_API_URL", "http://localhost:7860")
GPU_API = os.getenv("GPU_API_URL", "http://localhost:7862")
TITLE = "PII Redaction Model Arena"
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("PII_MAX_NEW_TOKENS", "256"))
FRONTEND_BUILD_ID = "frontend-2026-03-12-v2"
DATASET_URL = "https://huggingface.co/datasets/ai4privacy/pii-masking-200k"


def _call(api_url, text, model_path=None):
    t0 = time.perf_counter()
    payload = {"text": text, "max_new_tokens": DEFAULT_MAX_NEW_TOKENS}
    if model_path:
        payload["model_path"] = model_path
    r = requests.post(f"{api_url}/redact", json=payload, timeout=120)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.json().get("normalized", ""), latency_ms


def _gpu_available():
    try:
        r = requests.get(f"{GPU_API}/", timeout=3)
        r.raise_for_status()
        return True
    except Exception:
        return False


def _cpu_models():
    r = requests.get(f"{CPU_API}/models", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def _parse_selection(selection: str):
    if not selection or "::" not in selection:
        return "", ""
    return selection.split("::", 1)


def _selection_for_run(row, model_choices):
    if isinstance(model_choices, str):
        model_choices = json.loads(model_choices) if model_choices else []
    if row.get("model_type") == "hf_full":
        for c in model_choices:
            _label, value = _parse_selection(c)
            if value == "GPU_DEFAULT":
                return c
        return None

    target = os.path.basename((row.get("gguf_path") or "").strip())
    if not target:
        return None
    for c in model_choices:
        _label, value = _parse_selection(c)
        if value == "GPU_DEFAULT":
            continue
        if os.path.basename(value) == target:
            return c
    return None


def list_models():
    choices = []
    notes = []
    try:
        cpu_models = _cpu_models()
        for p in cpu_models:
            parent = os.path.basename(os.path.dirname(p))
            choices.append(f"CPU | {parent}/{os.path.basename(p)}::{p}")
        notes.append(f"CPU models: {len(cpu_models)}")
    except Exception as e:
        notes.append(f"CPU models unavailable: {e.__class__.__name__}")

    if _gpu_available():
        choices.append("GPU | merged_hf::GPU_DEFAULT")
        notes.append("GPU model: available")
    else:
        notes.append("GPU model: unavailable (run --profile gpu)")

    default_model = choices[0] if choices else None
    status = " | ".join(notes)
    return choices, default_model, status


def refresh_model_choices():
    models, default_model, status = list_models()
    alt_default = default_model
    if len(models) > 1:
        alt_default = models[1]
    return (
        gr.update(choices=models, value=default_model),
        gr.update(choices=models, value=alt_default),
        status,
        json.dumps(models),
    )


def _run_selected_model(selection: str, user_text: str):
    if "::" not in selection:
        raise ValueError("Invalid model selection format.")
    _, value = selection.split("::", 1)
    if value == "GPU_DEFAULT":
        return _call(GPU_API, user_text, model_path=None)
    return _call(CPU_API, user_text, model_path=value)


def _tag_count(text: str) -> int:
    return text.count("[")


def _arena_metrics(out1, out2, ms1, ms2, err):
    exact = "PENDING"
    sim = 0.0
    if out1 and out2:
        exact = "MATCH" if out1 == out2 else "DIFF"
        sim = SequenceMatcher(None, out1, out2).ratio()
    metrics = (
        f"- Exact output match: `{exact}`\n"
        f"- Similarity: `{sim:.4f}`\n"
        f"- Model 1 latency: `{ms1:.1f} ms`\n"
        f"- Model 2 latency: `{ms2:.1f} ms`\n"
        f"- Model 1 tags: `{_tag_count(out1) if out1 else 0}`\n"
        f"- Model 2 tags: `{_tag_count(out2) if out2 else 0}`"
    )
    if err:
        metrics += f"\n- Notes: `{err}`"
    return metrics


def arena_send(user_text, model1, model2, model1_hist, model2_hist):
    user_text = (user_text or "").strip()
    if not user_text:
        return "", model1_hist, model2_hist, "Enter input text."
    if not model1 or not model2:
        return "", model1_hist, model2_hist, "Select both models."

    model1_hist = (model1_hist or []) + [{"role": "user", "content": user_text}]
    model2_hist = (model2_hist or []) + [{"role": "user", "content": user_text}]

    err = None
    out1 = out2 = ""
    ms1 = ms2 = 0.0

    yield "", model1_hist, model2_hist, "Running both models asynchronously..."

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        pending = {
            ex.submit(_run_selected_model, model1, user_text): "m1",
            ex.submit(_run_selected_model, model2, user_text): "m2",
        }
        while pending:
            done, _ = concurrent.futures.wait(
                pending.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                key = pending.pop(fut)
                try:
                    out, ms = fut.result()
                except Exception as e:
                    msg = f"{'Model 1' if key == 'm1' else 'Model 2'} failed: {e.__class__.__name__}: {e}"
                    err = f"{err} | {msg}" if err else msg
                    out = f"[ERROR] {msg}"
                    ms = 0.0

                if key == "m1":
                    out1, ms1 = out, ms
                    model1_hist = model1_hist + [{"role": "assistant", "content": out1}]
                else:
                    out2, ms2 = out, ms
                    model2_hist = model2_hist + [{"role": "assistant", "content": out2}]

                yield (
                    "",
                    model1_hist,
                    model2_hist,
                    _arena_metrics(out1, out2, ms1, ms2, err),
                )


def arena_clear():
    return [], [], "Cleared."


def _api_json(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def _leaderboard_status(dataset, row_count: int) -> str:
    return (
        f"### Model Summary\n"
        f"- This leaderboard compares the quantized CPU-based fine-tuned LLM PII Redaction model variants against the GPU model.\n"
        f"- The foundation model is `mistralai/Mistral-7B-Instruct-v0.2`.\n"
        f"- The redaction model was fine-tuned for PII masking on `ai4privacy/pii-masking-200k`.\n"
    )


def load_leaderboard():
    try:
        payload = _api_json(f"{CPU_API}/eval/leaderboard")
    except Exception as e:
        return [], "[]", f"Failed to load leaderboard. Dataset: [ai4privacy/pii-masking-200k]({DATASET_URL})", [], "Failed to load leaderboard"

    rows = payload.get("rows", [])
    dataset = payload.get("dataset", "unknown")
    table = []
    for r in rows:
        table.append(
            [
                r.get("run_name"),
                r.get("model_type"),
                round(float(r.get("micro_f1") or 0.0), 4),
                round(float(r.get("macro_f1") or 0.0), 4),
                round(float(r.get("exact_match_rate") or 0.0), 4),
                round(float(r.get("avg_latency_ms") or 0.0), 1),
                round(float(r.get("p95_latency_ms") or 0.0), 1),
            ]
        )
    status = _leaderboard_status(dataset, len(rows))
    details = "No leaderboard rows available."
    per_tag = []
    if rows:
        row = rows[0]
        run_name = row.get("run_name")
        try:
            summary = _summary_for_run(run_name)
            per_tag = _per_tag_rows(summary, row)
            details = _detail_markdown(run_name, row, summary)
        except Exception as e:
            details = f"Failed to load top-run summary: {e}"
    return table, json.dumps(rows), status, per_tag, details


def _model_block(summary: dict, row: dict | None):
    if "models" in summary:
        if row and row.get("model_type") == "hf_full":
            return summary["models"].get("hf", {})
        return summary["models"].get("gguf", {})
    return summary.get("model", {})


def _summary_for_run(run_name: str):
    return _api_json(f"{CPU_API}/eval/summary/{run_name}")


def _detail_markdown(run_name: str, row: dict | None, summary: dict):
    m = _model_block(summary, row)
    return (
        f"### Run: `{run_name}`\n"
        f"- Type: `{(row or {}).get('model_type', 'n/a')}`\n"
        f"- Micro F1: `{(m.get('micro_f1') or 0.0):.4f}`\n"
        f"- Macro F1: `{(m.get('macro_f1') or 0.0):.4f}`\n"
        f"- Exact Match: `{(m.get('exact_match_rate') or 0.0):.4f}`\n"
        f"- Avg Latency: `{(m.get('avg_latency_ms') or 0.0):.1f} ms`\n"
        f"- P95 Latency: `{(m.get('p95_latency_ms') or 0.0):.1f} ms`\n"
        f"- Missing Tags: `{int(m.get('missing_tags') or 0)}`\n"
        f"- Spurious Tags: `{int(m.get('spurious_tags') or 0)}`"
    )


def _per_tag_rows(summary: dict, row: dict | None):
    m = _model_block(summary, row)
    per_tag = m.get("per_tag", {})
    rows = []
    for tag, vals in sorted(per_tag.items()):
        rows.append(
            [
                tag,
                round(float(vals.get("f1") or 0.0), 4),
                round(float(vals.get("precision") or 0.0), 4),
                round(float(vals.get("recall") or 0.0), 4),
                int(vals.get("tp") or 0),
                int(vals.get("fp") or 0),
                int(vals.get("fn") or 0),
            ]
        )
    return rows


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")

    with gr.Tab("Model Arena"):
        status = gr.Markdown("Load model versions, then compare outputs side-by-side.")
        refresh_models_btn = gr.Button("Refresh Model Versions")
        with gr.Row():
            left_model = gr.Dropdown(label="Model 1", choices=[])
            right_model = gr.Dropdown(label="Model 2", choices=[])
        with gr.Row():
            left_chat = gr.Chatbot(label="Model 1", height=300, type="messages")
            right_chat = gr.Chatbot(label="Model 2", height=300, type="messages")
        arena_msg = gr.Textbox(label="Message", lines=2, placeholder="Enter text to redact...")
        with gr.Row():
            send_btn = gr.Button("Send to Both")
            clear_btn = gr.Button("Clear Chats")
        arena_metrics = gr.Markdown("Metrics will appear here.")

    with gr.Tab("Leaderboard"):
        lb_status = gr.Markdown(
            "Leaderboard auto-loads from backend on page open. "
            f"Original dataset: [ai4privacy/pii-masking-200k]({DATASET_URL})"
        )
        leaderboard = gr.Dataframe(
            headers=[
                "run_name",
                "model_type",
                "micro_f1",
                "macro_f1",
                "exact_match_rate",
                "avg_latency_ms",
                "p95_latency_ms",
            ],
            label="Model Leaderboard",
        )
        details = gr.Markdown("Top-run details will appear here.")
        per_tag = gr.Dataframe(
            headers=["tag", "f1", "precision", "recall", "tp", "fp", "fn"],
            label="Per-tag Metrics",
        )

    leaderboard_rows_state = gr.Textbox(value="[]", visible=False)
    model_choices_state = gr.Textbox(value="[]", visible=False)

    refresh_models_btn.click(
        fn=refresh_model_choices,
        inputs=[],
        outputs=[left_model, right_model, status, model_choices_state],
        show_api=False,
    )
    send_btn.click(
        fn=arena_send,
        inputs=[arena_msg, left_model, right_model, left_chat, right_chat],
        outputs=[arena_msg, left_chat, right_chat, arena_metrics],
        show_api=False,
    )
    clear_btn.click(
        fn=arena_clear,
        inputs=[],
        outputs=[left_chat, right_chat, arena_metrics],
        show_api=False,
    )
    demo.load(
        fn=refresh_model_choices,
        inputs=[],
        outputs=[left_model, right_model, status, model_choices_state],
        show_api=False,
    )
    demo.load(
        fn=load_leaderboard,
        inputs=[],
        outputs=[leaderboard, leaderboard_rows_state, lb_status, per_tag, details],
        show_api=False,
    )

root_app = FastAPI()


@root_app.get("/manifest.json")
def web_manifest():
    return JSONResponse(
        {
            "name": TITLE,
            "short_name": "PII Arena",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#ffffff",
        }
    )


@root_app.get("/static/fonts/{font_path:path}")
def missing_font(font_path: str):
    # Browsers probe multiple fallback fonts; return 204 instead of noisy 404s.
    return Response(status_code=204)


@root_app.get("/")
def root_redirect():
    return RedirectResponse(url="/app", status_code=307)


app = gr.mount_gradio_app(root_app, demo, path="/app")


if __name__ == "__main__":
    print(f"Starting frontend build: {FRONTEND_BUILD_ID}", flush=True)
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
    )
