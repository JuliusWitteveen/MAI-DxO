from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements.txt"
ENV_FILE = PROJECT_ROOT / ".env"
ALT_ENV_FILE_1 = PROJECT_ROOT / "keys.env"
ALT_ENV_FILE_2 = PROJECT_ROOT / "env.local"


def ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])  # noqa: S603,S607

    # Upgrade core tooling quietly
    subprocess.check_call([  # noqa: S603,S607
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ])


def _pip_install(args: List[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])  # noqa: S603,S607


def install_requirements() -> None:
    if not REQ_FILE.exists():
        return

    # Install each requirement one-by-one so we can special-case 'swarms'
    lines = [ln.strip() for ln in REQ_FILE.read_text(encoding="utf-8").splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]

    is_windows = sys.platform.startswith("win")
    for req in reqs:
        try:
            if req.lower().startswith("swarms"):
                if is_windows:
                    # On Windows, skip dependency resolution to avoid uvloop
                    print("Installing 'swarms' without dependencies (Windows workaround)...")
                    _pip_install([req, "--no-deps"])  # type: ignore[arg-type]
                    minimal = [
                        "litellm>=1.48.0",
                        "httpx",
                        "aiohttp",
                        "aiofiles",
                        "numpy",
                        "networkx",
                        "openai",
                        "psutil",
                        "pypdf==5.1.0",
                        "python-dotenv",
                        "PyYAML",
                        "rich",
                        "schedule",
                        "tenacity",
                        "toml",
                        "docstring-parser==0.16",
                        "mcp",
                    ]
                    _pip_install(minimal)
                else:
                    _pip_install([req])
            else:
                _pip_install([req])
        except subprocess.CalledProcessError as e:
            # Continue on best-effort; print and move on so a single failing optional dep
            # (e.g., uvloop) doesn't block the prototype.
            print(f"Warning: could not install {req}: {e}")


def print_key_hint() -> None:
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    if not any(keys.values()):
        print("\n[Hint] No model API key detected. Set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY.")


def run_server() -> None:
    import uvicorn  # Imported after installation

    host = os.getenv("MAIDX_UI_HOST", "127.0.0.1")
    port = int(os.getenv("MAIDX_UI_PORT", "8000"))
    # Disable auto-reload by default on Windows to avoid optional deps
    reload_flag = os.getenv("MAIDX_UI_RELOAD", "0") not in {"0", "false", "False"}

    print(f"\nStarting MAI-DxO Web UI at http://{host}:{port}/\n")
    uvicorn.run("webui.server:app", host=host, port=port, reload=reload_flag)


def main(argv: List[str] | None = None) -> int:
    print("Ensuring Python packaging tools...")
    ensure_pip()
    print("Installing/Verifying requirements... (this may take a minute)")
    install_requirements()
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore

        for f in (ENV_FILE, ALT_ENV_FILE_1, ALT_ENV_FILE_2):
            if f.exists():
                load_dotenv(dotenv_path=f, override=False)
    except Exception:
        pass
    print_key_hint()
    run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


