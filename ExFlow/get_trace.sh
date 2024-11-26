num_experts=$1
echo "Running trace with $1 number of experts"

python3 ../src/start_harmony.py \
        --dataset bookcorpus \
        --num_samples 680 \
        --batch_size 85 \
        --seq_len 60 \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size 8 \
        --warmup_rounds 0 \
        --path "./raw_trace/$num_experts"