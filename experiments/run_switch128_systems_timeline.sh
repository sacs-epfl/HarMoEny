#!/bin/bash

datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=92160
seq_len=1024
num_experts=128
world_size=8
expert_cache_size=16
eq_tokens=1512
warmup_len=3

fastmoe_batch=18
fastermoe_batch=18
deepspeed_batch=2

cd ..
python3 src/start_fastmoe.py \
    --system_name fastmoe \
    --dataset "constant" \
    --num_samples $num_samples \
    --batch_size $fastmoe_batch \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --num_experts $num_experts \
    --world_size $world_size \
    --warmup_rounds $warmup_len \
    --router_num_experts_skewed 1 \
    --enable_router_random "True" \
    --pa "outputs/exp-switch128-systems-timeline/$datetime/fastmoe"

python3 src/start_fastmoe.py \
    --system_name fastermoe \
    --dataset "constant" \
    --num_samples $num_samples \
    --batch_size $fastmoe_batch \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --num_experts $num_experts \
    --world_size $world_size \
    --warmup_rounds $warmup_len \
    --router_num_experts_skewed 1 \
    --enable_router_random "True" \
    --pa "outputs/exp-switch128-systems-timeline/$datetime/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset "constant" \
    --num_samples $num_samples \
    --batch_size $deepspeed_batch \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --num_experts $num_experts \
    --world_size $world_size \
    --warmup_rounds $warmup_len \
    --router_num_experts_skewed 1 \
    --enable_router_random "True" \
    --pa "outputs/exp-switch128-systems-timeline/$datetime/deepspeed"
