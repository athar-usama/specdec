"""Command-line entry point: ``specdec {train,generate,benchmark}``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="specdec")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train the tokenizer and both models")
    gen = sub.add_parser("generate", help="watch speculative decoding generate text live")
    gen.add_argument("--prompt", default="The lighthouse")
    gen.add_argument("--tokens", type=int, default=120)
    sub.add_parser("benchmark", help="measure naive vs. speculative decoding throughput")
    args = parser.parse_args(argv)

    if args.command == "train":
        from .demos.train import main as run

        run()
    elif args.command == "generate":
        from .demos.live_generate import main as run

        run(prompt=args.prompt, n_tokens=args.tokens)
    elif args.command == "benchmark":
        from .demos.benchmark import main as run

        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
