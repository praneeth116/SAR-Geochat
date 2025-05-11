#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1.5-7b"
################## VICUNA ##################
#liuhaotian/llava-v1.5-7b

 deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0,1,2 geochat/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path MBZUAI/geochat-7B \
    --version $PROMPT_VERSION \
    --data_path /home/cvpr_ug_4/GeoChat/annotations.json \
    --image_folder   /home/cvpr_ug_4/SAR/single_channel_rgb \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /home/cvpr_ug_4/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/snapshots/5414da88308e4287a29f2e9609256458afb0a981/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /home/cvpr_ug_4/GeoChat/model_weights_vh \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 7000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 16 \
#    --report_to wandb

