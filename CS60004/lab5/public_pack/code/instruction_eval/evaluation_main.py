"""Utilities for evaluating instruction following. See the package README."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from instruction_eval import evaluation_lib


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_PREDICTION_DATA = flags.DEFINE_string(
    "prediction_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  inputs = evaluation_lib.read_prompt_list(_INPUT_DATA.value)
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      _PREDICTION_DATA.value)

  # get instruction following results
  for func, output_file_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
      (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
  ]:
    logging.info("Generating %s...", output_file_name)
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    output_file_name = os.path.join(
        _OUTPUT_DIR.value, output_file_name + ".jsonl"
    )
    evaluation_lib.write_outputs(output_file_name, outputs)
    logging.info("Generated: %s", output_file_name)

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    evaluation_lib.print_report(outputs)


if __name__ == "__main__":
  app.run(main)
