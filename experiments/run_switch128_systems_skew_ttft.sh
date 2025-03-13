datetime=$(date +"%Y-%m-%d_%H-%M")

seq_len=1024
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
enable_skew=True
num_experts_skewed=1
eq_tokens=1512
warmup_len=3

skews=(0.0 0.5 0.9)
batches=(64 32 18)
num_sampless=(15360 7680 4320)

cd ..
for skew_index in "${!skews[@]}"
do
    skew="${skews[$skew_index]}"
    batch_size="${batches[$skew_index]}"
    num_samples="${num_sampless[$skew_index]}"

    python3 src/start_harmony.py \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --scheduling_policy harmony \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-switch128-systems-skew-ttft/$datetime/$skew-harmony"

    python3 src/start_harmony.py \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --scheduling_policy exflow \
        --expert_cache_size $expert_cache_size \
        --expert_placement "ExFlow/placement/switch-exp${num_experts}-gpu${world_size}.json" \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-switch128-systems-skew-ttft/$datetime/$skew-exflow"

    python3 src/start_fastmoe.py \
        --system_name fastmoe \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-switch128-systems-skew-ttft/$datetime/$skew-fastmoe"

    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-switch128-systems-skew-ttft/$datetime/$skew-fastermoe"
done
