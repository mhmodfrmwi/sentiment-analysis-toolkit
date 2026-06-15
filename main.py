"""CLI entry points for the sentiment analysis toolkit."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment Analysis Toolkit")
    parser.add_argument(
        "command",
        choices=["gui", "api", "web"],
        help="gui: Tkinter desktop app, api: FastAPI server, web: Streamlit UI",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    args = parser.parse_args()

    if args.command == "gui":
        from app.gui import setup_gui

        setup_gui()
    elif args.command == "api":
        import uvicorn

        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.headless",
                "true",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                "8501",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
