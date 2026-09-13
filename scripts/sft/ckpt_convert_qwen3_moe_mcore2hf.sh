#!/bin/bash

if [ $# -lt 3 ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ $# -gt 3 ];
then
  echo "$0: Too many arguments: $@"
  exit 1
else
  echo "We got some argument(s)"
  echo "==========================="
  echo "Number of arguments.: $#"
  echo "Arg #1..............: $1" # path to original Hugging Face weights

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

  echo "Arg #2..............: ${DATA_URL}" # path to Megatron checkpoints
  echo "Arg #3..............: ${OUTPUT_DIR}" # path to Hugging Face weights
  echo "==========================="
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1

INPUT_DIR="${DATA_URL%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ${INPUT_DIR} \
    --save-dir ${OUTPUT_DIR} \
    --hf-cfg-dir $1 \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe