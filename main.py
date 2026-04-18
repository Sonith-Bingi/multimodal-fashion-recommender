"""
main.py — Universal entry point for the Multimodal Fashion Recommender

This script allows you to run the full pipeline (data prep, embedding, training, evaluation)
from the command line, without using the notebook. It uses the src/recommender modules.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recommender.cli import main as cli_main



def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Fashion Recommender: Universal Entry Point"
    )
    parser.add_argument(
        "command",
        choices=["check", "summary", "train", "evaluate"],
        help="Pipeline step to run: check, summary, train, evaluate",
    )
    args = parser.parse_args()

    # Dispatch to CLI logic (already implemented in src/recommender/cli.py)
    cli_main([args.command])

if __name__ == "__main__":
    main()
