#!/bin/bash
output_path="data/hyperparametres/eq_tokens/$(date +"%Y-%m-%d_%H-%M")"

dataset="wikitext"
num_samples=24000
batch_size=25
seq_len=120
world_size=8
num_experts=128

#eq_tokens=(1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 2500 5000 10000 15000)
eq_tokens=(1 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)

cd ..
for i in {0..15}
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
        --pa $output_path/${eq_tokens[i]}
done

