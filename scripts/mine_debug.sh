#!/bin/bash

set -e

echo "===== JOB START ====="
echo "HOSTNAME: $(hostname)"
echo "START TIME: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /nas2/data/wjddb1025/STOP

group=group2-2

dataset=direction_mcq
fps=3

DATA_PATH=/local_datasets/STOP/4combo_v5/syn_shape_simple/videos_12f
train_csv=/local_datasets/STOP/4combo_v5/syn_shape_simple/train.json
val_csv=/local_datasets/STOP/4combo_v5/syn_shape_simple/val.json
features_path=/local_datasets/STOP/4combo_v5/syn_shape_simple/videos_12f_compressed

pretrained_dir=/nas2/data/wjddb1025/STOP/models/pretrained

do_train=1
do_eval=0

pretrained_clip_name=ViT-B/32
lr=1e-3
coef_lr=5e-4
wd=0.2
epochs=2
optim=AdamW
max_words=32
max_frames=12
batch_size=16
batch_size_val=8
num_workers=0
n_display=10
precision=fp32

freeze_clip=1
time_embedding=0
shared_latent_space=linear

init_method='tcp://127.0.0.1:6010'

current_datetime=$(TZ="Asia/Seoul" date +"%Y-%m-%d-%H:%M:%S")
model_dir=logs/${current_datetime}_${dataset}_STOP

echo "The model dir is ${model_dir}"

python main.py \
    --task_type direction_mcq \
    --datatype ${dataset} \
    --do_train ${do_train} \
    --do_eval ${do_eval} \
    --num_thread_reader ${num_workers} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --batch_size_val ${batch_size_val} \
    --n_display ${n_display} \
    --train_csv ${train_csv} \
    --val_csv ${val_csv} \
    --features_path ${features_path} \
    --output_dir ${model_dir} \
    --optim ${optim} \
    --lr ${lr} \
    --coef_lr ${coef_lr} \
    --wd ${wd} \
    --max_words ${max_words} \
    --max_frames ${max_frames} \
    --feature_framerate ${fps} \
    --freeze_layer_num 12 \
    --slice_framepos 2 \
    --loose_type \
    --linear_patch 2d \
    --sim_header meanP \
    --pretrained_clip_name ${pretrained_clip_name} \
    --precision ${precision} \
    --init_method ${init_method} \
    --pretrained_dir ${pretrained_dir} \
    --freeze_clip ${freeze_clip} \
    --time_embedding ${time_embedding} \
    --shared_latent_space ${shared_latent_space} \
    --temporal_prompt ${group} \
    --use_traj_prompt 1 \
    --traj_prompt_scale 0.005 \
    --traj_prompt_dropout 0.0 \
    --traj_feat_type patch_mean

echo "Training Finished!!!"