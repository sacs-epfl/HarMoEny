cd ..
python3 src/start_fastermoe.py \
        --dataset "constant" \
        --num_samples 640 \
        --batch_size 64 \
        --seq_len 1024 \
        --model_name "google/switch-base-128" \
        --num_experts 128 \
        --world_size 8 \
        --enable_router_skew True \
        --router_skew 0.5 \
        --router_num_experts_skewed 1 \
        --enable_fastermoe True \
        --pa "outputs/dump/test"