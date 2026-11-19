#!/bin/sh

if [ $# -lt 5 ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ $# -gt 5 ];
then
  echo "$0: Too many arguments: $@"
  exit 1
else
  echo "We got some argument(s)"
  echo "==========================="
  echo "Number of arguments.: $#"
  echo "Arg #1..............: $1" # model path
  echo "Arg #2..............: $2" # the number of NPUs on a single node
  echo "Arg #3..............: $3" # max_num_seqs
  echo "Arg #4..............: $4" # max_model_len
  echo "Arg #5..............: $5" # gpu-memory-utilization
  echo "==========================="
fi

MODEL_PATH=$1
NPUS=$2
MAX_NUM_SEQS=$3
MAX_MODEL_LEN=$4
GPU_MEMORY_UTILIZATION=$5

# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

exec vllm serve ${MODEL_PATH} \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size ${NPUS} \
--enable-expert-parallel \
--served-model-name Qwen3-30B-A3B-Thinking-2507 \
--max-num-seqs ${MAX_NUM_SEQS} \
--max-model-len ${MAX_MODEL_LEN} \
--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
