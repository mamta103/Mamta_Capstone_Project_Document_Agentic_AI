"""Start the application: python start_app.py [ui|api] (default: ui)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "ui": ["-m", "streamlit", "run", "app/frontend/streamlit_app.py"],
    "api": ["-m", "uvicorn", "app.backend.main:app", "--port", "8000"],
}


def main(args: list[str]) -> int:
    target = args[0] if args else "ui"
    if len(args) > 1 or target not in COMMANDS:
        print("Usage: python start_app.py [ui|api]")
        return 2
    try:
        # Runs from the project root so .env, .streamlit/ and ./chroma_db resolve from any folder.
        return subprocess.run([sys.executable, *COMMANDS[target]], cwd=ROOT).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
