#!/bin/bash

MODEL_PATH=$1 # model path
NEED_RIGOR_VALIDATION=1
MAX_NUM_SEQS=$2 # 24/48
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"

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

set -e

DATA_URL="${DATA_URL%/}"  # path to test set
OUTPUT_DIR="${OUTPUT_DIR%/}"  # path to inference result

echo "DATA_URL=${DATA_URL}"
find ${DATA_URL} -type f

echo "OUTPUT_DIR=${OUTPUT_DIR}"
find ${OUTPUT_DIR} -type f

pip install datasets
pip install pyarrow==23.0.0

########################################
# 1. launch vLLM
########################################

bash ./gandolf/llms/online_inference_deployment_scripts/qwen3_30B_A3B_Thinking_2507.sh \
  ${MODEL_PATH} \
  ${MA_NUM_GPUS} \
  ${MAX_NUM_SEQS} \
  81920 \
  0.90 \
  > ${OUTPUT_DIR}/vllm.log 2>&1 &

VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"

########################################
# 2. cleanup when shutting down
########################################

cleanup() {
    echo "Stopping vLLM..."

    if ps -p ${VLLM_PID} > /dev/null 2>&1; then
        kill ${VLLM_PID}
        wait ${VLLM_PID} || true
    fi
}

trap cleanup EXIT

########################################
# 3. waiting for vllm server
########################################

echo "Waiting for vLLM..."

for i in $(seq 1 900)
do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1
    then
        echo "vLLM is ready"
        break
    fi

    sleep 2
done

########################################
# 4. timeout check
########################################

if ! curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1
then
    echo "vLLM startup timeout"
    exit 1
fi

########################################
# 5. run inference
########################################

VALIDATION_ARG=""
if [ "${NEED_RIGOR_VALIDATION}" -eq 1 ]; then
    VALIDATION_ARG="--need_rigor_validation"
fi

python ./gandolf/inference/binary_classification.py \
	--input_path ${DATA_URL}/R2E_Gym.parquet \
	--output_dir ${OUTPUT_DIR}/R2E_Gym \
	--model_name Qwen3-30B-A3B-Thinking-2507 \
	--batch_size 200 \
	--max_output_tokens 12288 \
	--url http://127.0.0.1:8000/v1/completions \
	--concurrency ${MAX_NUM_SEQS} \
	${VALIDATION_ARG} \
	--is_thinking

echo "/R2E_Gym finished"
	
python ./gandolf/inference/binary_classification.py \
	--input_path ${DATA_URL}/SWE_Rebench.parquet \
	--output_dir ${OUTPUT_DIR}/SWE_Rebench \
	--model_name Qwen3-30B-A3B-Thinking-2507 \
	--batch_size 200 \
	--max_output_tokens 12288 \
	--url http://127.0.0.1:8000/v1/completions \
	--concurrency ${MAX_NUM_SEQS} \
	${VALIDATION_ARG} \
	--is_thinking

echo "/SWE_Rebench finished"

python ./gandolf/inference/binary_classification.py \
	--input_path ${DATA_URL}/SWE_Smith.parquet \
	--output_dir ${OUTPUT_DIR}/SWE_Smith \
	--model_name Qwen3-30B-A3B-Thinking-2507 \
	--batch_size 200 \
	--max_output_tokens 12288 \
	--url http://127.0.0.1:8000/v1/completions \
	--concurrency ${MAX_NUM_SEQS} \
	${VALIDATION_ARG} \
	--is_thinking

echo "/SWE_Smith finished"
