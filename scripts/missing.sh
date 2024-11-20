dataset=bookcorpus
num_samples=1040000

cd ..
python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size 8 \