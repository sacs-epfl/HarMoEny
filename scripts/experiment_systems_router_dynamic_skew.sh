dataset=constant
seq_len=120
num_experts=128
world_size=8
num_iters=250
output_path="data/systems/router_dynamic_skew/$(date +"%Y-%m-%d_%H-%M")"

cd ..
python3 src/start_harmony.py \
    --dataset $dataset \
    --num_samples $((2500 * $world_size * $num_iters)) \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --enable_router_skew True \
    --random_router_skew True \
    --pa $output_path/harmony

# python3 src/start_harmony.py \
#     --dataset $dataset \
#     --num_samples $((1000 * $world_size * $num_iters)) \
#     --batch_size 1000 \
#     --seq_len $seq_len \
#     --model_name "google/switch-base-$num_experts" \
#     --scheduling_policy "exflow" \
#     --expert_cache_size 16 \
#     --world_size $world_size \
#     --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
#     --random_router_skew True \
#     --pa $output_path/exflow

# python3 src/start_fastmoe.py \
#     --dataset $dataset \
#     --num_samples $((1000 * $world_size * $num_iters)) \
#     --batch_size 1000 \
#     --seq_len $seq_len \
#     --num_experts $num_experts \
#     --world_size $world_size \
#     --random_router_skew True \
#     --pa $output_path/fastmoe

# python3 src/start_fastermoe.py \
#     --dataset $dataset \
#     --num_samples $((1000 * $world_size * $num_iters)) \
#     --batch_size 1000 \
#     --seq_len $seq_len \
#     --num_experts $num_experts \
#     --world_size $world_size \
#     --random_router_skew True \
#     --pa $output_path/fastermoe

# deepspeed --num_gpus $world_size src/start_deepspeed.py \
#     --dataset $dataset \
#     --num_samples $((200 * $world_size * $num_iters)) \
#     --batch_size 200 \
#     --seq_len $seq_len \
#     --num_experts $num_experts \
#     --world_size $world_size \
#     --random_router_skew True \
#     --pa $output_path/deepspeed


#   --capacity_factor 64.0 \
