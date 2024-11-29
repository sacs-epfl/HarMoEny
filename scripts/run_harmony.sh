dataset=cocktail
num_samples=200000
num_experts=128
num_gpus=8
policy="harmony"
seq_len=120
batch_size=2140

cd ..
python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy $policy \
        --expert_cache_size $(($num_experts / $num_gpus)) \
        --world_size $num_gpus \
        --batch_size $batch_size \
        --path "outputs/harmony/run"
