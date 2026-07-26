# bash scripts/0-start_vllm.sh
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='0'

# vllm serve \
#   --model models/Qwen3.5-9B \
#   --served-model-name qwen-3.5-9b \
#   --port 12345 \
#   --tensor-parallel-size 1 \
#   --reasoning-parser qwen3 \
#   --language-model-only \
#   --gpu-memory-utilization 0.9 \
#   --trust_remote_code \
#   --enforce-eager

# vllm serve \
#   --model models/Qwen2.5-1.5B-Instruct \
#   --served-model-name qwen2.5-1.5b-instruct \
#   --port 12345 \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.9 \
#   --trust_remote_code \
#   --enforce-eager


vllm serve \
  --model /root/autodl-tmp/CS60004/lab1/outputs/task3_rft_merged \
  --served-model-name task3-rft \
  --port 12345 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --trust_remote_code \
  --enforce-eager