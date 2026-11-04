"""Run the MoodLens FastAPI server.

Usage:
    uv run python main.py
    uv run python main.py --reload           # auto-restart on code changes
    uv run python main.py --host 0.0.0.0 --port 9000
                                        # 0.0.0.0 exposes the API to your LAN
"""

import argparse

import uvicorn
from src.deployment.app import ALLOWED_ORIGINS


def main():
    parser = argparse.ArgumentParser(description="Run the MoodLens API server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 - local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-restart the server when source files change (development)",
    )
    args = parser.parse_args()

    if not ALLOWED_ORIGINS:
        print(
            "CORS is disabled (no browser origins allowed). "
            "Set MOODLENS_ALLOWED_ORIGINS to enable browser clients."
        )

    # String import path so --reload can re-import the app in fresh processes
    uvicorn.run(
        "src.deployment.app:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
