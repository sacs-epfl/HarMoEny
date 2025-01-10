datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=12800
seq_len=1024
world_size=8
num_experts=128
enable_skew=True
num_experts_skewed=1
warmup_len=3

skews=(0.0 0.5 0.9)

fastmoe_batches=(64 32 16)
fastermoe_batches=(64 32 16)

cd ..
for skew in "${skews[@]}"
do
   python3 src/start_fastmoe.py \
        --system_name fastmoe \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size ${fastmoe_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-switch128-systems-skew/$datetime/$skew-fastmoe"
    
    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size ${fastmoe_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-switch128-systems-skew/$datetime/$skew-fastermoe"
done