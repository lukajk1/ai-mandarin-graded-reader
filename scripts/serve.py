"""Start llama.cpp's llama-server with a local GGUF model.

    uv run python scripts/serve.py                     # GPU (Vulkan), first model in models/
    uv run python scripts/serve.py --cpu --threads 2   # rehearse a 2-vCPU host (HF CPU Basic)

Expects the llama.cpp release in tools/llama.cpp/ and models in models/ (both gitignored).
The server speaks the OpenAI API at http://127.0.0.1:<port>/v1, which `mandarin` uses by default.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = (
    ROOT
    / "tools"
    / "llama.cpp"
    / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=Path, help="GGUF file (default: first in models/)")
    parser.add_argument("--port", type=int, default=8081)
    # Prompts carry the whole known-word list (~6k tokens for HSK 1-3) plus a revision turn.
    parser.add_argument("--ctx", type=int, default=16384, help="context window in tokens")
    parser.add_argument("--cpu", action="store_true", help="no GPU offload")
    parser.add_argument("--threads", type=int, help="CPU threads (default: llama.cpp decides)")
    args = parser.parse_args()

    model = args.model or next(iter(sorted((ROOT / "models").glob("*.gguf"))), None)
    if model is None or not model.exists():
        sys.exit("no model found: put a .gguf file in models/ or pass --model")
    if not SERVER.exists():
        sys.exit(f"llama-server not found at {SERVER}")

    cmd = [
        str(SERVER),
        "--model", str(model),
        "--port", str(args.port),
        "--ctx-size", str(args.ctx),
        "--n-gpu-layers", "0" if args.cpu else "99",
        "--parallel", "1",
        "--jinja",
    ]  # fmt: skip
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    print(" ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
