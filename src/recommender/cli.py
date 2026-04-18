from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from .config import Settings
from .logging_utils import configure_logging
from .pipeline import RecommenderPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reco", description="Recommender project CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate expected artifacts and paths")
    sub.add_parser("summary", help="Print a lightweight pipeline summary")
    sub.add_parser("train", help="Run full training pipeline")
    sub.add_parser("evaluate", help="Run full evaluation pipeline")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    settings = Settings()
    pipeline = RecommenderPipeline(settings)

    if args.command == "check":
        status = pipeline.validate_artifacts()
        ok = status.catalog and status.index and status.vectors
        print(json.dumps(status.__dict__, indent=2))
        return 0 if ok else 1

    if args.command == "summary":
        print(json.dumps(pipeline.summarize_pipeline(), indent=2))
        return 0

    if args.command == "train":
        print(json.dumps(pipeline.train(), indent=2))
        return 0

    if args.command == "evaluate":
        metrics = pipeline.evaluate()
        print(
            json.dumps(
                {
                    "recall_at_5": metrics.recall_at_5,
                    "recall_at_10": metrics.recall_at_10,
                    "mrr_at_10": metrics.mrr_at_10,
                },
                indent=2,
            )
        )
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
