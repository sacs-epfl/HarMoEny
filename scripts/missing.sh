dataset=bookcorpus
num_samples=10000
output_path="outputs"

echo $output_path

cd ..
python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 2000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/harmonymoe"
