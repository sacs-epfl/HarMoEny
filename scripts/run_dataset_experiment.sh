#!/bin/bash

datasets=("wikitext" "bookcorpus" "random" "sst2" "wmt10")
num_samples=450000

date_str=$(date +"%Y-%m-%d")
output_path="outputs/$date_str"

cd ..
for dataset in "${datasets[@]}" 
do 
    python3 start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5550 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "adnexus" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/$dataset"
    
    python3 start.py \
        --system_name fastmoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 3250 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --world_size 8 \
        --pa "$output_path/$dataset"
    
    deepspeed --num_gpus 8 start.py \
        --system_name deepspeed-inference \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 850 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --pa "$output_path/$dataset"
done 