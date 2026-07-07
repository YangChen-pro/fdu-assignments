from __future__ import annotations

import argparse
from .data import summarize_dataset
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["splitA", "splitB", "splitC", "splitD"])
    args = parser.parse_args()
    summary = summarize_dataset(args.data_root, args.splits)
    write_json(args.output, summary)
    print(summary)


if __name__ == "__main__":
    main()
