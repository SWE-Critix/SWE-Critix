#!/bin/bash

if [ $# -lt 6 ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ $# -gt 6 ];
then
  echo "$0: Too many arguments: $@"
  exit 1
else
  echo "We got some argument(s)"
  echo "==========================="
  echo "Number of arguments.: $#"
  echo "Arg #1..............: $1" # TP
  echo "Arg #2..............: $2" # PP
  echo "Arg #3..............: $3" # EP
  echo "Arg #4..............: $4" # ETP (MUST be 1 or the TP value）

  for arg in "$@"
  do
      case $arg in
          --data_url=*)
              DATA_URL="${arg#*=}"
              ;;
          --output_dir=*)
              OUTPUT_DIR="${arg#*=}"
              ;;
      esac
  done

  echo "Arg #5..............: ${DATA_URL}" # path to Hugging Face weights
  echo "Arg #6..............: ${OUTPUT_DIR}" # path to Megatron checkpoints
  echo "==========================="
fi


export CUDA_DEVICE_MAX_CONNECTIONS=1

INPUT_DIR="${DATA_URL%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

ETP_ARG=""
if [ "$4" -eq 1 ]; then
    ETP_ARG="--expert-tensor-parallel-size 1"
fi

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size $1 \
    --target-pipeline-parallel-size $2 \
    --target-expert-parallel-size $3 \
    ${ETP_ARG} \
    --load-dir ${INPUT_DIR} \
    --save-dir ${OUTPUT_DIR} \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe