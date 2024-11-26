dataset=bookcorpus
num_samples=200000

cd ..
python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size 8 \
        --path "outputs/harmony"

#         --batch_size 5000 \
