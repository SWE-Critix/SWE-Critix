EXP_NAME=experiment_name # NEED CONFIG
MODEL_PATH=/path/to/sft_checkpoint # NEED CONFIG
MCORE_MODEL_PATH=/path/to/Megatron_weight # NEED CONFIG
CKPTS_DIR=/path/to/checkpoints # NEED CONFIG
TRAIN_FILE=/path/to/all_together.parquet # NEED CONFIG
TEST_FILE=/path/to/test.parquet # NEED CONFIG (since test_freq=-1 in this script, test.parquet can be any mocked data that conforms to VeRL's required format)
REWARD_FILE=/path/to/SWE-Critix/scripts/rl/reward.py # NEED CONFIG
TOTAL_EPOCHS=1
SAVE_FREQ=100

# Project Configuration
project_name='GRPO-Qwen3-30B-A3B-CodeForge-Preview-Patch-Correctness'
echo "${project_name}:${EXP_NAME} starts!"

# Node Info
NNODES=${MA_NUM_HOSTS}
NPUS_PER_NODE=${MA_NUM_GPUS}

ROOT_CKPTS_DIR=${CKPTS_DIR}
CKPTS_DIR=${CKPTS_DIR}/${project_name}/${EXP_NAME}
echo "MODEL_PATH=${MODEL_PATH}"
echo "MCORE_MODEL_PATH=${MCORE_MODEL_PATH}"
echo "CKPTS_DIR=${CKPTS_DIR}"

echo "TRAIN_FILE=${TRAIN_FILE}"
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS}"

# Data Configuration
max_prompt_length=$((1024 * 68))
max_response_length=$((1024 * 60))

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# Training Batch Configuration
train_prompt_bsz=32
n_resp_per_prompt=8
train_prompt_mini_bsz=32  # data.train_batch_size must be >= actor.ppo_mini_batch_size

# Performance and Memory Related Configuration
all_offload=True
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 8))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 8))
optimizer_offload_fraction=1

# Megatron Configuration
train_tp=2
train_pp=8
train_ep=16
train_etp=1
train_cp=8

# vLLM Configuration
gen_tp=4
gen_dp=4
gen_ep=16
gpu_memory_utilization=0.7
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=8192  # lower down to 4096 or 2048 if OOM

# Data Configuration
DATA_ARGS=(
    # File Paths
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    # Data Structure
    data.prompt_key=prompt
    # Batch and Length Configuration
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    # Preprocessing
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=32
    data.truncation='error'
)

# Model Configuration
MODEL_ARGS=(
    # Model Path
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # Model Processing
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# RL Algorithm Configuration
ALGORITHM_ARGS=(
    # Advantage Estimation
    algorithm.adv_estimator=${adv_estimator}
    # KL Divergence Control
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# Actor Model Configuration
ACTOR_ARGS=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss Function Configuration
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.lr=1e-6
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
    # Megatron Parallelism Strategy
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload}
    # Model Weights Management
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
    # Transformer Architecture Optimizations
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${train_cp}
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.position_embedding_type=rope
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
)

# Reference Model Configuration
REF_ARGS=(
    # Core Runtime Settings
    actor_rollout_ref.ref.use_torch_compile=False
    # Log Probability Inference
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Megatron Parallelism Strategy
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    actor_rollout_ref.ref.megatron.use_distributed_optimizer=True
    # Model Weights Management
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
    # Transformer Architecture Optimizations
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel=True
)

# Rollout Configuration
ROLLOUT_ARGS=(
    # Rollout Engine
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    # Generation Parameters，lower down max_num_seqs to 32 or 16 if OOM
    actor_rollout_ref.rollout.max_num_seqs=64
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    # Log Probability Inference
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Memory Management
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    # Parallelism Strategy
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    # Performance Optimization
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
)

# Trainer Configuration
TRAINER_ARGS=(
    # Logger Configuration
    trainer.logger='["console"]'
    # Project Settings
    trainer.project_name="${project_name}"
    trainer.experiment_name="${EXP_NAME}"
    # Hardware Configuration
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    # Training Schedule
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=${SAVE_FREQ}
    # Checkpoint Directory
    trainer.default_local_dir="${CKPTS_DIR}"
)

# Reward Configuration
REWARD_ARGS=(
    reward.custom_reward_function.path=${REWARD_FILE}
)

# Resumption Configuration
RESUME_ARGS=(
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model', 'optimizer', 'extra']"
)

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@" 2>&1 | tee ${ROOT_CKPTS_DIR}/${project_name}_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log