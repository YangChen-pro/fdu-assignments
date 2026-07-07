#!/usr/bin/env bash
set -euo pipefail
UPLOAD=0
if [[ "${1:-}" == "--upload" ]]; then
  UPLOAD=1
elif [[ "${1:-}" == "--no-upload" || -z "${1:-}" ]]; then
  UPLOAD=0
else
  echo "usage: bash hw3/task2/scripts/run_all.sh [--no-upload|--upload]" >&2
  exit 2
fi
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
cd "$TASK2_ROOT"
bash hw3/task2/scripts/dry_run.sh
bash hw3/task2/scripts/train.sh hw3/task2/configs/act_splitA.yaml 8
bash hw3/task2/scripts/train.sh hw3/task2/configs/act_splitABC.yaml 8
bash hw3/task2/scripts/evaluate.sh hw3/task2/configs/act_splitA.yaml hw3/task2/outputs/act_splitA/checkpoints/best.pt act_splitA
bash hw3/task2/scripts/evaluate.sh hw3/task2/configs/act_splitABC.yaml hw3/task2/outputs/act_splitABC/checkpoints/best.pt act_splitABC
bash hw3/task2/scripts/evaluate.sh hw3/task2/configs/act_splitA.yaml hw3/task2/outputs/act_splitA/checkpoints/final.pt act_splitA_final
bash hw3/task2/scripts/evaluate.sh hw3/task2/configs/act_splitABC.yaml hw3/task2/outputs/act_splitABC/checkpoints/final.pt act_splitABC_final
bash hw3/task2/scripts/evaluate_breakdown.sh hw3/task2/configs/act_splitA.yaml hw3/task2/outputs/act_splitA/checkpoints/final.pt act_splitA_final
bash hw3/task2/scripts/evaluate_breakdown.sh hw3/task2/configs/act_splitABC.yaml hw3/task2/outputs/act_splitABC/checkpoints/final.pt act_splitABC_final
bash hw3/task2/scripts/build_results.sh
if [[ "$UPLOAD" == "1" ]]; then
  bash hw3/task2/scripts/upload_modelscope.sh hw3/task2/outputs/act_splitA youngchen/CS60003 hw3/task2/outputs/act_splitA/modelscope_upload.json hw3/task2
  bash hw3/task2/scripts/upload_modelscope.sh hw3/task2/outputs/act_splitABC youngchen/CS60003 hw3/task2/outputs/act_splitABC/modelscope_upload.json hw3/task2
fi
