###################################################
# @Author: Yingying Shi
# @Email:  yyshi@buaa.edu.cn
# @Date: 2026-10-29
# @Description: Script of SemGuard.
# This Software is free for non-commercial, research and evaluation use.
###################################################
EXP_NAME=${DECODER_BASE_NAME}_msg${msg_len}_${num_epochs}_${type_fusion_weight}
###################################################
# Stage 1: Train the watermark decoder
# msg_len | num_epochs
# 4 8 16 ->  400
# 32 48  ->  800
# for example
python make_decoder_divid.py \
    --mode train \
    --msg_len $msg_len \
    --save \
    --num_epochs $num_epochs \
    --sdir decoders/nerf_synthetic/${EXP_NAME} \
    --type_fusion_weight $type_fusion_weight \  # continuous
    --type_fusion_weight_max 0.9 \
    --type_fusion_weight_min 0.0

###################################################
# Stage 2: Train and evaluate the SemGuard 4DGS model(see https://github.com/hustvl/4DGaussians in details)
# this script is an example for NeRF Synthetic data
root=<your_nerf_dir>
out_root=<your_render_dir>
exp_root=<your_Watermark_output_dir>
item=<nerf_item>
mdir=$out_root/$item
sdir=$exp_root/${EXP_NAME}/$item
python run_4Ddivid.py \
        --decoder_path decoders/nerf_synthetic/${EXP_NAME} \
        -s "$root/$item" \
        -m "$mdir" \
        -w \
        --mode train \
        --msg_len "$msg_len" \
        --sdir "$sdir" \
        --batch_size 16 \
        --dtype blender \
        --num_epochs2 "$num_epochs2"
###################################################

###################################################
# view and evaluate the watermark decoder
# this script is an example for NeRF Synthetic data

resume_num=<your_resume_num>
python run_4Ddivid.py \
        --decoder_path decoders/nerf_synthetic/${EXP_NAME} \
        -s "$root/$item" \
        -m "$mdir" \
        -w \
        --mode eval \
        --msg_len "$msg_len" \
        --sdir "$sdir" \
        --batch_size 16 \
        --dtype blender \
        --num_epochs2 100 \
        --test_cams 10 \
        --resume_num "$resume_num" \
        --save_map