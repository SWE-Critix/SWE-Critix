#!/bin/bash

if [ $# -lt 12 ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ $# -gt 12 ];
then
  echo "$0: Too many arguments: $@"
  exit 1
else
  echo "We got some argument(s)"
  echo "==========================="
  echo "Number of arguments.: $#"
  echo "Arg #1..............: $1" # path to tokenizer
  echo "Arg #2..............: $2" # TP
  echo "Arg #3..............: $3" # PP
  echo "Arg #4..............: $4" # EP
  echo "Arg #5..............: $5" # ETP
  echo "Arg #6..............: $6" # CP
  echo "Arg #7..............: $7" # Globe Batch Size
  echo "Arg #8..............: $8" # Train Iterations 
  echo "Arg #9..............: $9" # Sequence Length
  echo "Arg #10..............: ${10}" # Save Interval

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

  echo "Arg #11..............: ${DATA_URL}" # path to tokenized dataset
  echo "Arg #12..............: ${OUTPUT_DIR}" # path to megatron checkpoints
  echo "NPUS_PER_NODE = ${MA_NUM_GPUS}"
  echo "MASTER_ADDR = ${VC_WORKER_HOSTS%%,*}"
  echo "NNODES = ${MA_NUM_HOSTS}"
  echo "NODE_RANK = ${VC_TASK_INDEX}"
  echo "==========================="
fi


export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=5400
export HCCL_IF_BASE_PORT=48600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1


NPUS_PER_NODE=${MA_NUM_GPUS}
MASTER_ADDR=${VC_WORKER_HOSTS%%,*}
MASTER_PORT=6000
NNODES=${MA_NUM_HOSTS}
NODE_RANK=${VC_TASK_INDEX}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_LOAD_DIR="${OUTPUT_DIR}"
CKPT_SAVE_DIR="${OUTPUT_DIR}"
DATA_PATH="${DATA_URL%/}/alpaca"
TOKENIZER_PATH="$1"

TP=$2
PP=$3
EP=$4
ETP=$5
CP=$6
CP_TYPE='ulysses_cp_algo'
SEQ_LENGTH=$9
GBS=$7
TRAIN_ITERS=$8
SAVE_INTERVAL=${10}

LATEST_CKPT_FILE="${CKPT_SAVE_DIR%/}/latest_checkpointed_iteration.txt"

RESUME_TRAINING=false

if [ -f "$LATEST_CKPT_FILE" ]; then
    LAST_ITER=$(cat "$LATEST_CKPT_FILE" | tr -d '[:space:]')
    echo "Found latest checkpoint iteration: $LAST_ITER"

    if [ "$LAST_ITER" -gt 1 ]; then
        RESUME_TRAINING=true
        echo "Resume training detected."
    else
        echo "Initial training detected (iteration = 1)."
    fi
else
    echo "Checkpoint file not found."
    exit 1
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --expert-tensor-parallel-size ${ETP} \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 768 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-permute-fusion \
    --moe-alltoall-overlap-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type aux_loss \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --recompute-activation-function \
    --moe-zero-memory level1
"

TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --lr 1.25e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --manual-gc \
    --manual-gc-interval 50
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE}
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --prompt-type qwen3 \
    --tokenizer-type PretrainedFromHF \
    --enable-thinking true \
    --no-pad-to-seq-lengths \
    --pad-to-multiple-of 128
"


if [ "$RESUME_TRAINING" = true ]; then
    OUTPUT_ARGS="
        --log-interval 1 \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${SAVE_INTERVAL} \
        --eval-iters 0 \
        --load ${CKPT_LOAD_DIR} \
        --save ${CKPT_SAVE_DIR} \
        --log-throughput
    "
else
    OUTPUT_ARGS="
        --log-interval 1 \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${SAVE_INTERVAL} \
        --eval-iters 0 \
        --no-load-optim \
        --load ${CKPT_LOAD_DIR} \
        --save ${CKPT_SAVE_DIR} \
        --no-load-rng \
        --log-throughput
    "
fi


if [ "$RESUME_TRAINING" = true ]; then
    TUNE_ARGS="
        --stage sft \
        --is-instruction-dataset \
        --tokenizer-not-use-fast
    "
else
    TUNE_ARGS="
        --finetune \
        --stage sft \
        --is-instruction-dataset \
        --tokenizer-not-use-fast
    "
fi


torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl

