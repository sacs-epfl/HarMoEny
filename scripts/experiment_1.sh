dataset=bookcorpus
num_samples=1000000
output_path="outputs/experiment_1/$(date +"%Y-%m-%d_%H-%M")"

echo $output_path

cd ..
python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 3000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size 8 \
        --time_dense true \
        --pa "$output_path/deepspeed"

python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 3000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "adnexus" \
        --expert_cache_size 16 \
        --world_size 8 \
        --time_dense true \
        --pa "$output_path/adnexus"