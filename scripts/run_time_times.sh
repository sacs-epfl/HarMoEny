dataset=constant
num_samples=200000
num_experts=128
num_gpus=8
seq_len=120

cd ..
# python3 src/start_harmony.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --seq_len $seq_len \
#         --model_name "google/switch-base-$num_experts" \
#         --scheduling_policy "deepspeed" \
#         --expert_cache_size 16 \
#         --world_size $num_gpus \
#         --batch_size 2000 \
#         --enable_router_skew True \
#         --router_skew 0.5 \
#         --path "outputs/timetime/harmony_no_rebalancing_router_skew_50"

python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size $num_gpus \
        --batch_size 2000 \
        --enable_router_skew True \
        --router_skew 0.5 \
        --router_num_experts_skew 10 \
        --path "outputs/timetime/multi/harmony_router_skew50_min"

# python3 src/start_harmony.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --seq_len $seq_len \
#         --model_name "google/switch-base-$num_experts" \
#         --scheduling_policy "harmony" \
#         --expert_cache_size 16 \
#         --world_size $num_gpus \
#         --batch_size 2000 \
#         --enable_router_skew True \
#         --router_skew 0.5 \
#         --disable_async_fetch True \
#         --router_num_experts_skew 10 \
#         --path "outputs/timetime/multi/harmony_no_async_fetch_router_skew50"



