#!/usr/bin/env python
"""Probe which NVIDIA-hosted chat models are alive and return usable JSON.

Run this before trusting any hard-coded model id. Availability on
build.nvidia.com changes without notice -- during development of this project,
two Qwen models went from working to HTTP 410 "end of life" in a single day.

    uv run python scripts/probe_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from agentic_rag.nvidia import chat  # noqa: E402

CANDIDATES = [
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-120b",
    "google/gemma-4-31b-it",
    "meta/llama-4-maverick-17b-128e-instruct",
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
]

PROMPT = 'Return only this JSON and nothing else: {"ok": true, "n": 7}'


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    print(f"{'model':<45} {'status':<10} {'ms':>6}  json?")
    print("-" * 78)
    alive: list[str] = []
    for model in CANDIDATES:
        t0 = time.time()
        try:
            out = chat(model, [{"role": "user", "content": PROMPT}],
                       max_tokens=60, timeout=45.0)
            ms = int((time.time() - t0) * 1000)
            try:
                parsed = json.loads(out.strip().strip("`").removeprefix("json").strip())
                ok = parsed.get("ok") is True
                print(f"{model:<45} {'OK':<10} {ms:>6}  {'yes' if ok else 'malformed'}")
                if ok:
                    alive.append(model)
            except Exception:
                print(f"{model:<45} {'OK':<10} {ms:>6}  no ({out[:28]!r})")
        except Exception as exc:
            ms = int((time.time() - t0) * 1000)
            msg = str(exc)
            status = "410 EOL" if "410" in msg else "404" if "404" in msg else \
                     "timeout" if "timeout" in msg.lower() else "error"
            print(f"{model:<45} {status:<10} {ms:>6}  -")

    print(f"\n{len(alive)} models returned clean JSON:")
    for m in alive:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
