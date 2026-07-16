#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def preferred_python():
    root = Path(__file__).resolve().parent
    candidates = [
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    candidate = preferred_python()
    if candidate and Path(sys.executable).resolve() != candidate.resolve():
        os.execv(
            str(candidate),
            [str(candidate), str(Path(__file__).with_name("server.py")), *sys.argv[1:]],
        )
    from server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
