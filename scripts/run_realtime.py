"""Launch the integrated PHANTOM browser application on loopback."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run visibly simulated demo mode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("PHANTOM's local UI must be bound to a loopback address")
    if args.demo:
        os.environ["PHANTOM_CONFIG"] = str(REPOSITORY_ROOT / "configs" / "realtime_demo.yaml")
    else:
        os.environ.setdefault(
            "PHANTOM_CONFIG", str(REPOSITORY_ROOT / "configs" / "realtime_cpu.yaml")
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the API dependencies before launching PHANTOM") from exc
    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run("app.realtime:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
