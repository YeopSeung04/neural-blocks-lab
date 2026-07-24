#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def preferred_python():
    root = Path(__file__).resolve().parent
    for candidate in (
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return candidate
    return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python_runner.py <script> [args...]")
    candidate = preferred_python()
    candidate_environment = candidate.parent.parent if candidate else None
    if (
        candidate
        and Path(sys.prefix).resolve() != candidate_environment.resolve()
    ):
        os.execv(str(candidate), [str(candidate), *sys.argv[1:]])
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
