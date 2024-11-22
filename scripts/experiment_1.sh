dataset=bookcorpus
num_samples=20000
output_path="data/experiment_1/$(date +"%Y-%m-%d_%H-%M")"

echo $output_path

cd ..
python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 250 \
        --seq_len 60 \
        --model_name "google/switch-base-32" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 4 \
        --world_size 8 \
        --path "$output_path/deepspeed_policy"

python3 src/start_t5_base.py \
        --dataset $dataset \
        --num_samples $(($num_samples / 8)) \
        --batch_size 250 \
        --seq_len 60 \
        --path "$output_path/dense"

python3 src/start_t5_param_amt_match.py \
        --num_experts 32 \
        --dataset $dataset \
        --num_samples $(($num_samples / 8)) \
        --batch_size 250 \
        --seq_len 60 \
        --path "$output_path/dense_param_match" 
