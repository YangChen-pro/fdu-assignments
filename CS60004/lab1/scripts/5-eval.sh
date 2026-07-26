# bash scripts/5-eval.sh

python scripts/4-eval.py \
  --input_path datasets/provide_to_students/val.jsonl \
  --model_name task3-rft \
  --plan A B C \
  --output_root outputs/task4
