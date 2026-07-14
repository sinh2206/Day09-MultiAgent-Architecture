from __future__ import annotations

import argparse
from pathlib import Path

from app.graph import ShoppingAssistant
from app.utils import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student scaffold CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output-dir", default="src/artifacts/batch")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        summary = assistant.run_batch(
            Path(args.test_file),
            Path(args.output_dir),
            rebuild_index=args.rebuild_index,
        )
        print(dump_json(summary))
        return
    if not args.question:
        parser.error("Use --question or --batch")
    result = assistant.ask(
        args.question,
        trace_file=Path(args.trace_file) if args.trace_file else None,
        rebuild_index=args.rebuild_index,
    )
    print(result["final_answer"])


if __name__ == "__main__":
    main()
