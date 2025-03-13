#!/bin/bash

datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=92160
seq_len=1024
num_experts=128
world_size=8
expert_cache_size=16
eq_tokens=1512
warmup_len=3

policies=("deepspeed" "harmony" "even_split" "exflow")
policies=("harmony")
batch_size_balanced=64
batch_size_unbalanced=18

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
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --scheduling_policy $policy \
        --expert_cache_size $expert_cache_size \
        --expert_placement "ExFlow/placement/switch-exp${num_experts}-gpu${world_size}.json" \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --router_num_experts_skewed 1 \
        --enable_router_random "True" \
        --pa "outputs/exp-switch128-policy-timeline/$datetime/$policy"
done
