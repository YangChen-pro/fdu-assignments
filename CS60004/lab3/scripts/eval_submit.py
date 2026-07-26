import argparse
import json
from pathlib import Path

from preprocess_utils import load_countdown_data, summarize_outputs, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="本地评测提交文件")
    parser.add_argument("--test-file", default="lab3/datasets/rl_train_test_datasets/raw_test.json")
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    test_rows = load_countdown_data(args.test_file)
    test_by_id = {row["id"]: row for row in test_rows}
    rows = []
    with Path(args.pred_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sample = test_by_id[item["id"]]
            rows.append(
                {
                    "id": item["id"],
                    "numbers": sample["numbers"],
                    "target": sample["target"],
                    "prediction": item["prediction"],
                    "output_tokens": item.get("output_tokens", 0),
                }
            )
    summary = summarize_outputs(rows)
    summary["prediction_path"] = args.pred_file
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
