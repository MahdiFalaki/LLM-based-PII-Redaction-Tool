# src/pii_masking/train/axolotl_train.py
import os
import subprocess
import importlib.util
from pathlib import Path
from huggingface_hub import login

# project root: .../src/pii_masking/train -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))
ax_src = os.getenv("AXOLOTL_SRC")
if ax_src:
    os.environ["PYTHONPATH"] = f"{ax_src}:{os.environ['PYTHONPATH']}"

def _has_axolotl_cli() -> bool:
    return importlib.util.find_spec("axolotl.cli.preprocess") is not None

def _resolve_cfg() -> Path:
    c = PROJECT_ROOT / "config" / "pii_config.yml"
    if c.exists():
        return c
    else:
        raise FileNotFoundError("pii_config.yml not found under configs/ or config/")

def main():
    cfg = _resolve_cfg()
    if not _has_axolotl_cli():
        raise RuntimeError(
            "axolotl CLI module not found. Install a version that provides "
            "'axolotl.cli.preprocess' and 'axolotl.cli.train' (or run from an Axolotl source checkout)."
        )

    # optional HF login (env only — no hardcoded secrets)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("🔐 Logging into Hugging Face Hub…")
        login(token=hf_token)
    else:
        print("ℹ️  HF_TOKEN not set; skipping HF login.")

    print("🔄 Axolotl preprocess…")
    subprocess.run(
        ["python", "-m", "axolotl.cli.preprocess", str(cfg)],
        check=True,
    )

    print("🚀 Axolotl train…")
    subprocess.run(
        ["accelerate", "launch", "-m", "axolotl.cli.train", str(cfg)],
        check=True,
    )

if __name__ == "__main__":
    main()
