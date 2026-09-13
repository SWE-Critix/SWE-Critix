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
  echo "Arg #1..............: $1" # path to tokenizer

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

  echo "Arg #2..............: ${DATA_URL}" # path to alpaca-style dataset, which can be a single file or a directory. Supporting .parquet/.csv/.json/.jsonl/.txt/.arrow
  echo "Arg #3..............: ${OUTPUT_DIR}" # path to output dir
  echo "==========================="
fi

INPUT_PATH="${DATA_URL%/}"
TOKENIZER_PATH=$1
OUTPUT_FOLDER_PATH="${OUTPUT_DIR%/}/alpaca"

python preprocess_data.py \
    --input "${INPUT_PATH}" \
    --tokenizer-name-or-path "${TOKENIZER_PATH%/}" \
    --output-prefix ${OUTPUT_FOLDER_PATH} \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --enable-thinking true \
    --map-keys '{"system":"system","prompt":"instruction","query":"input","response":"output"}' \
    --prompt-type qwen3