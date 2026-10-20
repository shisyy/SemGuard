##########################
# msg_len | num_epochs
# 4 8 16 ->  100
# 32 48  ->  200
# 64 72  ->  300
##########################

# #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
cd $(dirname $0)

###########################################
#exp info
DECODER_BASE_NAME=gard

msg_len=16  # 16|32|48
num_epochs=100

EXP_NAME=${DECODER_BASE_NAME}_msg${msg_len}_${num_epochs}

# #stagte1: 训练水印解码器
# python make_decoder_divid.py \
#     --mode train \
#     --msg_len $msg_len \
#     --save \
#     --num_epochs $num_epochs \
#     --sdir decoders/dnerf/${EXP_NAME} \
#     --type_fusion_weight $type_fusion_weight \
#     --type_fusion_weight_max 0.9 \
#     --type_fusion_weight_min 0.0



#stage2: 加水印
export USED_NUM_FRAMES=8
root=4DGaussians/data/dnerf
out_root=pretrained/dnerf
exp_root=outputs/dnerf


# 按顺序跑的 8 个数据集
items=(
  bouncingballs
  hellwarrior
  hook
  jumpingjacks
  lego
  mutant
  standup
  trex
)

declare -A RESUME_MAP=(
  [bouncingballs]=100
  [hellwarrior]=100
  [hook]=100
  [jumpingjacks]=100
  [lego]=80
  [mutant]=90
  [standup]=90
  [trex]=100
)

num_epochs2=100
for item in "${items[@]}"; do
    mdir="$out_root/$item"
    sdir="$exp_root/${EXP_NAME}/8fame/$item"
    cfg="4DGaussians/arguments/dnerf/${item}.py"
    resume_num=${RESUME_MAP[$item]}

    echo "=============================="
    echo "Running item=$item"
    echo "model_path=$mdir"
    echo "configs=$cfg"
    echo "sdir=$sdir"
    echo "=============================="

    python run_divid_view.py \
        --decoder_path decoders/dnerf/${EXP_NAME} \
        --source_path "$root/$item" \
        --model_path "$mdir" \
        --mode eval \
        --msg_len "$msg_len" \
        --sdir "$sdir" \
        --batch_size 8 \
        --configs "$cfg" \
        --dataset_type blender \
        --train_num_views_per_time_real 100 \
        --test_num_views_per_time 10 \
        --train_num_frame_real "$USED_NUM_FRAMES" \
        --test_num_frame "$USED_NUM_FRAMES" \
        --num_epochs2 "$num_epochs2" \
        --resume_num "$resume_num" \
        --save_map
done

#######################################################################################

##########################################
#exp info
DECODER_BASE_NAME=gard

msg_len=16  # 16|32|48
num_epochs=100

EXP_NAME=${DECODER_BASE_NAME}_msg${msg_len}_${num_epochs}

# #stagte1: 训练水印解码器
# python make_decoder_divid_.py \
#     --mode eval \
#     --msg_len $msg_len \
#     --save \
#     --num_epochs $num_epochs \
#     --sdir decoders/nerf_llff_data/${EXP_NAME} \
#     --type_fusion_weight $type_fusion_weight \
#     --type_fusion_weight_max 0.9 \
#     --type_fusion_weight_min 0.0


#stage2: 加水印
export USED_NUM_FRAMES=8
root=4DGaussians/data/hypernerf/virg
out_root=pretrained/hypernerf
exp_root=outputs/hypernerf


# 按顺序跑的 4 个数据集
items=(
  3dprinter
  banana
  broom2
  chicken
)

declare -A RESUME_MAP=(
  [3dprinter]=10
  [banana]=30
  [broom2]=30
  [chicken]=20
)
num_epochs2=100
for item in "${items[@]}"; do
    mdir="$out_root/$item"
    sdir="$exp_root/${EXP_NAME}/8fame/$item"
    cfg="4DGaussians/arguments/hypernerf/${item}.py"
    resume_num=${RESUME_MAP[$item]}

    echo "=============================="
    echo "Running item=$item"
    echo "model_path=$mdir"
    echo "configs=$cfg"
    echo "sdir=$sdir"
    echo "=============================="

    python run_divid_view.py \
        --decoder_path decoders/hypernerf/${EXP_NAME} \
        --source_path "$root/$item" \
        --model_path "$mdir" \
        --mode eval \
        --msg_len "$msg_len" \
        --sdir "$sdir" \
        --batch_size 8 \
        --configs "$cfg" \
        --dataset_type nerfies \
        --train_num_views_per_time_real 100 \
        --test_num_views_per_time 10 \
        --train_num_frame_real "$USED_NUM_FRAMES" \
        --test_num_frame "$USED_NUM_FRAMES" \
        --num_epochs2 "$num_epochs2" \
        --resume_num "$resume_num" \
        --save_map
done
        
        
     
  