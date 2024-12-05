#!/bin/bash
output_path="data/hyperparametres/eq_tokens/$(date +"%Y-%m-%d_%H-%M")"

dataset="wikitext"
num_samples=10000
batch_size=2500
seq_len=120
world_size=8
num_experts=128

eq_tokens=(0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250)

cd ..
for i in {0..25}
do
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --eq_tokens ${eq_tokens[i]} \
        --pa $output_path/${eq_tokens[i]}/harmony
done