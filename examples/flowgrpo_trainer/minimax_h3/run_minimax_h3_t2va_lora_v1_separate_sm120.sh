#!/usr/bin/env bash
# MiniMax H3 T2VA LoRA FlowGRPO (V1 separate_async) with the standalone
# vLLM-Omni rollout on sm120 Blackwell GPUs and the FSDP Actor on Hopper GPUs.
#
#   Actor   : _flash_3_varlen_hub (kernels-hub FA3, Hopper)
#   Rollout : FLASH_ATTN -> CuTe FlashAttention-4 (flash_attn.cute) on sm120
#
# Model, LoRA, reward, pipeline and CPS FlowGRPO knobs come from
# run_minimax_h3_t2va_lora_v1.sh; this wrapper only switches to separate_async
# and selects the attention pair. Install flash-attn-4 on the rollout node(s)
# and run check_sm120_flash_attn.py there first. The Ray cluster must place the
# trainer pool on the Hopper node and the rollout pool on the sm120 node; see
# "Standalone rollout on sm120 GPUs" in README.md.
#
#   TRAIN_GPUS            GPUs for the Actor (+ reward models) on the Hopper node
#   ROLLOUT_NNODES        sm120 nodes for the standalone rollout
#   ROLLOUT_GPUS_PER_NODE sm120 GPUs per rollout node (multiple of ROLLOUT_TP)
#   ASYNC                 1: rollout overlaps training (<= 1 policy version stale)
#                         0: sync_compatible, algorithmically on-policy
#   BYPASS                True: reuse rollout log-probs as old log-probs
#                         (algorithm.rollout_correction.bypass_mode)
set -euo pipefail

TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ASYNC=${ASYNC:-1}
BYPASS=${BYPASS:-True}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}

if ((ROLLOUT_GPUS_PER_NODE % ROLLOUT_TP != 0)); then
  echo "ROLLOUT_GPUS_PER_NODE ($ROLLOUT_GPUS_PER_NODE) must be a multiple of ROLLOUT_TP ($ROLLOUT_TP)." >&2
  exit 1
fi
if ((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0)); then
  echo "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be a multiple of PPO_MINI_BATCH_SIZE ($PPO_MINI_BATCH_SIZE)." >&2
  exit 1
fi
# separate_async requires train_batch_size == parameter_sync_step * ppo_mini_batch_size.
PARAM_SYNC_STEP=$((TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE))

if [[ "$ASYNC" == 1 ]]; then
  async_args=(
    trainer.v1.separate_async.sync_compatible=False
    trainer.v1.separate_async.num_warmup_batches=1
    trainer.v1.sampler.max_off_policy_threshold=2
  )
else
  async_args=(
    trainer.v1.separate_async.sync_compatible=True
    trainer.v1.separate_async.num_warmup_batches=0
    trainer.v1.sampler.max_off_policy_threshold=1
  )
fi

script_dir=$(dirname "$(readlink -f "$0")")

# NUM_GPUS sizes the Actor pool (and the colocated agent workers) in the base recipe.
NUM_GPUS=$TRAIN_GPUS ROLLOUT_TP=$ROLLOUT_TP \
  bash "$script_dir/run_minimax_h3_t2va_lora_v1.sh" \
  actor_rollout_ref.model.attn_backend=_flash_3_varlen_hub \
  actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=$PARAM_SYNC_STEP \
  trainer.v1.sampler.max_off_policy_strategy=drop \
  "${async_args[@]}" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.rollout.nnodes=$ROLLOUT_NNODES \
  actor_rollout_ref.rollout.n_gpus_per_node=$ROLLOUT_GPUS_PER_NODE \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.trust_remote_code=True \
  algorithm.rollout_correction.bypass_mode=$BYPASS \
  trainer.experiment_name=minimax_h3_t2va_lora_v1_separate_sm120 \
  "$@"
