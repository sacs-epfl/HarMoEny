#!/bin/bash

datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=34560
seq_len=1024
world_size=6
expert_cache_size=10
eq_tokens=1512
warmup_len=3

policies=("deepspeed" "harmony" "even_split" "exflow")
policies=("exflow")
batch_size_balanced=32
batch_size_unbalanced=4

cd ..
for policy in "${policies[@]}"
do
    if [[ "$policy" == "harmony" || "$policy" == "even_split" ]]; then
        batch_size=$batch_size_balanced
    else
        batch_size=$batch_size_unbalanced
    fi

    python3 src/start_harmony.py \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        --loader transformers \
        --d_model 2048 \
        --num_experts 60 \
        --type_moe_parent Qwen2MoeDecoderLayer \
        --type_moe Qwen2MoeSparseMoeBlock \
        --router_tensor_path gate \
        --name_experts experts \
        --scheduling_policy $policy \
        --expert_cache_size $expert_cache_size \
        --expert_placement "ExFlow/placement/qwen-gpu${world_size}.json" \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --router_num_experts_skewed 1 \
        --enable_router_random "True" \
        --pa "outputs/exp-qwen-policy-timeline/$datetime/$policy"
done
