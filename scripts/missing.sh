dataset=bookcorpus
num_samples=1000000
output_path="outputs"

echo $output_path

cd ..
python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5250 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "adnexus" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/harmonymoe"