from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from utils import read_jsonl, write_json, write_jsonl


def validate_records(role_rows: list[dict], instruction_rows: list[dict]) -> dict:
    role_ids = [row.get("id") for row in role_rows]
    instruction_keys = [row.get("key") for row in instruction_rows]
    return {
        "role_count": len(role_rows),
        "instruction_count": len(instruction_rows),
        "role_unique_ids": len(set(role_ids)),
        "instruction_unique_keys": len(set(instruction_keys)),
        "role_keys": sorted(role_rows[0]) if role_rows else [],
        "instruction_keys": sorted(instruction_rows[0]) if instruction_rows else [],
        "role_has_required_fields": all({"id", "character_id", "prediction"} <= set(row) for row in role_rows),
        "instruction_has_required_fields": all({"key", "prediction"} <= set(row) for row in instruction_rows),
    }


def write_submission_zip(output_dir: Path) -> Path:
    zip_path = output_dir / "task_5_test.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(output_dir / "role_interview_predictions.jsonl", "role_interview_predictions.jsonl")
        zf.write(output_dir / "instruction_test_predictions.jsonl", "instruction_test_predictions.jsonl")
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lab5 Task5 submission zip from model prediction files.")
    parser.add_argument("--role-predictions", type=Path, required=True)
    parser.add_argument("--instruction-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/submission_task5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    role_rows = read_jsonl(args.role_predictions)
    instruction_rows = read_jsonl(args.instruction_predictions)

    role_output = args.output_dir / "role_interview_predictions.jsonl"
    instruction_output = args.output_dir / "instruction_test_predictions.jsonl"
    write_jsonl(role_output, role_rows)
    write_jsonl(instruction_output, instruction_rows)
    zip_path = write_submission_zip(args.output_dir)

    manifest = validate_records(role_rows, instruction_rows)
    manifest.update(
        {
            "zip_file": str(zip_path),
            "role_source": str(args.role_predictions),
            "instruction_source": str(args.instruction_predictions),
            "postprocess": False,
        }
    )
    write_json(args.output_dir / "submission_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
