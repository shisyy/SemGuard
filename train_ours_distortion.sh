# 第一个数据
export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")"
#!/usr/bin/env bash
set -e

root=4DGaussians/data/dnerf
out_root=pretrained/dnerf
exp_root=outputs/dnerf

msg_len=16
export USED_NUM_FRAMES=8

## combined distortions
atypes=(
  Crop
  Resize
  Rotation
  Brightness
  JPEGCompress
  VAEBmshj2018
)

# 按顺序跑的 8 个数据集
items=(
  # bouncingballs
  # hellwarrior
  # hook
  # jumpingjacks
  # lego
  mutant
  standup
  trex
)

for item in "${items[@]}"; do
  mdir="$out_root/$item"
  sdir="$exp_root/watermark_ours_Distortion_${msg_len}/8fame/$item"
  cfg="4DGaussians/arguments/dnerf/${item}.py"

  echo "=============================="
  echo "Running item=$item"
  echo "model_path=$mdir"
  echo "configs=$cfg"
  echo "sdir=$sdir"
  echo "=============================="

  python run_watermark_ours.py \
    --source_path "$root/$item" \
    --model_path "$mdir" \
    --mode train \
    --msg_len "$msg_len" \
    --sdir "$sdir" \
    --batch_size 8 \
    --configs "$cfg" \
    --dataset_type blender \
    --train_num_views_per_time_real 100 \
    --test_num_views_per_time 100 \
    --train_num_frame_real "$USED_NUM_FRAMES" \
    --test_num_frame "$USED_NUM_FRAMES" \
    --num_epochs 100 \
    --atypes "${atypes[@]}"
done







