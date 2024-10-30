dataset=arxiver
num_samples=450000
output_path="outputs"

cd ..
python3 start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 200 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size 8 \
        --time_dense true \
        --pa $output_path