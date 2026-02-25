"""Start the FastAPI prediction server.

Usage:
    python scripts/serve.py [--port 8000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start churn prediction API server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Churn Prediction API on {args.host}:{args.port}")
    print(f"Docs: http://localhost:{args.port}/docs")

    uvicorn.run(
        "src.serving.api:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
