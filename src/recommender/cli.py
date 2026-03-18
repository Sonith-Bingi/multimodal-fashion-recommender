from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

from .config import Settings
from .logging_utils import configure_logging
from .pipeline import RecommenderPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reco", description="Recommender project CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate expected artifacts and paths")
    sub.add_parser("summary", help="Print a lightweight notebook summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    settings = Settings()
    pipeline = RecommenderPipeline(settings)

    if args.command == "check":
        status = pipeline.validate_artifacts()
        ok = status.notebook and status.index and status.vectors
        print(json.dumps(status.__dict__, indent=2))
        return 0 if ok else 1

    if args.command == "summary":
        print(json.dumps(pipeline.summarize_notebook(), indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
