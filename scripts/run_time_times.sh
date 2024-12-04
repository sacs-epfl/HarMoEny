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
#         --expert_cache_size $(($num_experts / $num_gpus)) \
#         --world_size $num_gpus \
#         --batch_size 1000 \
#         --enable_router_skew True \
#         --router_skew 0.5 \
#         --path "outputs/timetime/deepspeed"

python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size $(($num_experts / $num_gpus)) \
        --world_size $num_gpus \
        --batch_size 2500 \
        --enable_router_skew True \
        --router_skew 0.5 \
        --path "outputs/timetime/harmony"



