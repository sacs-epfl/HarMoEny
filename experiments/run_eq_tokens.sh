datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=5120
seq_len=1024
batch_size=64
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
warmup_len=3
eq_tokens=(16 32 64 128 256 512 1024 2048 4096 8192 16384)

cd ..
for eq_token in "${eq_tokens[@]}"
do
    python3 src/start_harmony.py \
            --dataset "wikitext" \
            --num_samples $num_samples \
            --batch_size $batch_size \
            --seq_len $seq_len \
            --model_name "google/switch-base-$num_experts" \
            --num_experts $num_experts \
            --scheduling_policy harmony \
            --expert_cache_size $expert_cache_size \
            --world_size $world_size \
            --eq_tokens $eq_token \
            --expert_fetching_strategy "async-cpu" \
            --warmup_rounds $warmup_len \
            --pa "outputs/exp-eq-tokens/$datetime/$eq_token"
done