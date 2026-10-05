#!/usr/bin/env python3
"""Entry point. From the repo root:

    python run.py all
    python run.py data|gan|detect|infer|serve|ui|agent-eval
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# CPU is enough for this MLP. Silence missing-CUDA noise.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="WGAN-GP credit-card fraud pipeline (TensorFlow).")
    p.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "data", "gan", "detect", "infer", "serve", "ui", "agent-eval"],
        help="which stage to run (default: all)",
    )
    p.add_argument("--csv", type=str, default=None, help="for infer: path to a CSV of raw transactions")
    p.add_argument("--values", type=str, default=None, help="for infer: comma-separated raw feature row")
    args = p.parse_args(argv)

    if args.stage == "agent-eval":
        from src.agent_eval import main as eval_main

        return eval_main()

    if args.stage in ("all", "data"):
        from src.data import prepare

        prepare()

    if args.stage in ("all", "gan"):
        from src.train_gan import train as train_gan
        from src.generate import generate

        train_gan()
        generate()

    if args.stage in ("all", "detect"):
        from src.train_detector import train as train_detect

        train_detect()

    if args.stage == "infer":
        from src.infer import main as infer_main

        infer_argv = []
        if args.csv:
            infer_argv += ["--csv", args.csv]
        if args.values:
            infer_argv += ["--values", args.values]
        return infer_main(infer_argv)

    if args.stage == "serve":
        import uvicorn

        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
        return 0

    if args.stage == "ui":
        import subprocess

        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "ui.py"), "--server.port", "8501"]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
