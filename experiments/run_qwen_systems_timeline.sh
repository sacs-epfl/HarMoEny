#!/bin/bash

datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=5760
seq_len=1024
world_size=6
num_experts_skewed=1
warmup_len=3

fastmoe_batch=4
fastermoe_batch=4
deepspeed_batch=1

cd ..
python3 src/start_fastmoe.py \
    --system_name fastmoe \
    --dataset "constant" \
    --num_samples $num_samples \
    --batch_size $fastmoe_batch \
    --seq_len $seq_len \
    --num_experts 60 \
    --d_model 2048 \
    --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    --type_moe_parent Qwen2MoeDecoderLayer \
    --type_moe Qwen2MoeSparseMoeBlock \
    --router_tensor_path gate \
    --name_experts experts \
    --world_size $world_size \
    --warmup_rounds $warmup_len \
    --router_num_experts_skewed 1 \
    --enable_router_random "True" \
    --pa "outputs/exp-qwen-systems-timeline/$datetime/fastmoe"

python3 src/start_fastmoe.py \
    --system_name fastermoe \
    --dataset "constant" \
    --num_samples $num_samples \
    --batch_size $fastmoe_batch \
    --seq_len $seq_len \
    --num_experts 60 \
    --d_model 2048 \
    --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    --type_moe_parent Qwen2MoeDecoderLayer \
    --type_moe Qwen2MoeSparseMoeBlock \
    --router_tensor_path gate \
    --name_experts experts \
    --world_size $world_size \
    --warmup_rounds $warmup_len \
    --router_num_experts_skewed 1 \
    --enable_router_random "True" \
    --pa "outputs/exp-qwen-systems-timeline/$datetime/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset "constant" \
    --num_samples $num_samples \
    --batch_size $deepspeed_batch \
    --seq_len $seq_len \
    --num_experts 60 \
    --d_model 2048 \
    --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    --type_moe_parent Qwen2MoeDecoderLayer \
    --type_moe Qwen2MoeSparseMoeBlock \
    --router_tensor_path gate \
    --name_experts experts \
    --world_size $world_size \
    --warmup_rounds $warmup_len \
    --router_num_experts_skewed 1 \
    --enable_router_random "True" \
    --pa "outputs/exp-qwen-systems-timeline/$datetime/deepspeed"
