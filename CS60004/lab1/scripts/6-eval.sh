# bash scripts/6-eval.sh

python scripts/4-eval.py \
  --input_path outputs/6-golden_answer_300.jsonl \
  --model_name /root/autodl-tmp/CS60004/lab1/outputs/task2/v5-0324223414/latest \
  --plan A B C \
  --output_root outputs/task6


# 用 qwen-3.5 评测时候，temp 和 top_p 设置不同
# python scripts/4-eval.py \
#   --input_path datasets/provide_to_students/train.jsonl \
#   --model_name qwen-3.5-9b \
#   --temperature 1.0 \
#   --max_tokens 8192 \
#   --num_workers 16 \
#   --plan B \
#   --output_root outputs/task6
