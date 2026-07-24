#!/usr/bin/env python3
import argparse
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
    candidate_environment = candidate.parent.parent if candidate else None
    if (
        candidate
        and Path(sys.prefix).resolve() != candidate_environment.resolve()
    ):
        os.execv(
            str(candidate),
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    parser = argparse.ArgumentParser(description="Neural Blocks Lab server")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NBL_WEB_WORKERS", "1")),
    )
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.legacy:
        from server import BACKEND, NeuralBlocksHandler, ThreadingHTTPServer

        server = ThreadingHTTPServer((args.bind, args.port), NeuralBlocksHandler)
        print(f"Neural Blocks legacy server: http://{args.bind}:{args.port}")
        print(f"Classroom database: {BACKEND.database.description}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    import uvicorn

    uvicorn.run(
        "asgi_app:app",
        host=args.bind,
        port=args.port,
        workers=max(1, args.workers),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
