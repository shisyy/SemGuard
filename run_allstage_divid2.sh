#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USED_NUM_FRAMES="${USED_NUM_FRAMES:-8}"

DECODER_BASE_NAME="${DECODER_BASE_NAME:-divid}"
DECODER_EPOCHS="${DECODER_EPOCHS:-800}"
EMBED_EPOCHS="${EMBED_EPOCHS:-100}"
FUSION_SCHEDULE="${FUSION_SCHEDULE:-continuous}"
FUSION_MIN="${FUSION_MIN:-0.0}"
FUSION_MAX="${FUSION_MAX:-0.9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

DNERF_DATA_ROOT="${DNERF_DATA_ROOT:-${PROJECT_ROOT}/4DGaussians/data/dnerf}"
DNERF_MODEL_ROOT="${DNERF_MODEL_ROOT:-${PROJECT_ROOT}/pretrained/dnerf}"
HYPERNERF_DATA_ROOT="${HYPERNERF_DATA_ROOT:-${PROJECT_ROOT}/4DGaussians/data/hypernerf/virg}"
HYPERNERF_MODEL_ROOT="${HYPERNERF_MODEL_ROOT:-${PROJECT_ROOT}/pretrained/hypernerf}"

MESSAGE_LENGTHS=(16 32 48)
DNERF_SCENES=(
  bouncingballs
  hellwarrior
  hook
  jumpingjacks
  lego
  mutant
  standup
  trex
)
HYPERNERF_SCENES=(3dprinter banana broom2 chicken)

train_decoder() {
  local benchmark="$1"
  local msg_len="$2"
  local exp_name="${DECODER_BASE_NAME}_msg${msg_len}_${DECODER_EPOCHS}_${FUSION_SCHEDULE}"

  python make_decoder_divid.py \
    --mode train \
    --msg_len "$msg_len" \
    --save \
    --num_epochs "$DECODER_EPOCHS" \
    --sdir "decoders/${benchmark}/${exp_name}" \
    --type_fusion_weight "$FUSION_SCHEDULE" \
    --type_fusion_weight_min "$FUSION_MIN" \
    --type_fusion_weight_max "$FUSION_MAX"
}

run_benchmark() {
  local benchmark="$1"
  local data_root="$2"
  local model_root="$3"
  local config_group="$4"
  local dataset_type="$5"
  shift 5
  local scenes=("$@")

  for msg_len in "${MESSAGE_LENGTHS[@]}"; do
    local exp_name="${DECODER_BASE_NAME}_msg${msg_len}_${DECODER_EPOCHS}_${FUSION_SCHEDULE}"
    train_decoder "$benchmark" "$msg_len"

    for scene in "${scenes[@]}"; do
      local model_path="${model_root}/${scene}"
      local source_path="${data_root}/${scene}"
      local config_path="4DGaussians/arguments/${config_group}/${scene}.py"
      local save_dir="${OUTPUT_ROOT}/${benchmark}/${exp_name}/8frame/${scene}"

      echo "============================================================"
      echo "benchmark=${benchmark} scene=${scene} message_length=${msg_len}"
      echo "source_path=${source_path}"
      echo "model_path=${model_path}"
      echo "save_dir=${save_dir}"
      echo "============================================================"

      python run_divid.py \
        --decoder_path "decoders/${benchmark}/${exp_name}" \
        --source_path "$source_path" \
        --model_path "$model_path" \
        --mode train \
        --msg_len "$msg_len" \
        --sdir "$save_dir" \
        --batch_size 8 \
        --configs "$config_path" \
        --dataset_type "$dataset_type" \
        --train_num_views_per_time_real 100 \
        --test_num_views_per_time 100 \
        --train_num_frame_real "$USED_NUM_FRAMES" \
        --test_num_frame "$USED_NUM_FRAMES" \
        --num_epochs2 "$EMBED_EPOCHS"
    done
  done
}

if [[ "${RUN_DNERF:-1}" == "1" ]]; then
  run_benchmark \
    dnerf \
    "$DNERF_DATA_ROOT" \
    "$DNERF_MODEL_ROOT" \
    dnerf \
    blender \
    "${DNERF_SCENES[@]}"
fi

if [[ "${RUN_HYPERNERF:-1}" == "1" ]]; then
  run_benchmark \
    hypernerf \
    "$HYPERNERF_DATA_ROOT" \
    "$HYPERNERF_MODEL_ROOT" \
    hypernerf \
    nerfies \
    "${HYPERNERF_SCENES[@]}"
fi
