pkill -9 python
ray stop --force
rm -rf /tmp/ray
rm -rf /home/ma-user/.cache/torch_extensions/

DEFAULT_SH="/path/to/SWE-Critix/scripts/rl/run_qwen3-30b-a3b_128k_grpo_megatron_vllm_npu.sh" # NEED CONFIG

export ACL_DEVICE_SYNC_TIMEOUT=7200
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=5400
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DISABLE_L2_CACHE=1
export VLLM_USE_V1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:1024
export MULTI_STREAM_MEMORY_REUSE=1
export CPU_AFFINITY_CONF=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0

ulimit -n 32768

NNODES=${MA_NUM_HOSTS}
echo "NNODES=${NNODES}"
NPUS_PER_NODE=${MA_NUM_GPUS}
echo "NPUS_PER_NODE=${NPUS_PER_NODE}"

MASTER_HOST=$(echo $VC_WORKER_HOSTS | cut -d ',' -f1)
MASTER_ADDR=$(getent hosts $MASTER_HOST | awk '{print $1}')
echo "MASTER_IP = ${MASTER_ADDR}"

NIC=eth0
echo "NIC=${NIC}"

export HCCL_SOCKET_IFNAME="${NIC}"
export GLOO_SOCKET_IFNAME="${NIC}"

CURRENT_IP=${MA_CURRENT_IP}
echo "CURRENT_IP = ${MA_CURRENT_IP}"

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # launch master node
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # check if device_count == NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count nodes (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES nodes. Current node count: $device_count"
          sleep 5
      fi
  done
else
  # child nodes try to connect to the master node
  while true; do
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600