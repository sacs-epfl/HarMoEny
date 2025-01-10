datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=12800
seq_len=1024
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
enable_skew=True
num_experts_skewed=1
enable_random=False
enable_uniform=False
eq_tokens=1512 # 1512
warmup_len=3

skews=(0.0 0.5 0.9)
policies=("deepspeed" "harmony" "drop" "even_split" "exflow")
policies=("harmony")

batch_size_harmony_drop_even_split=64
batch_size_deepspeed_exflow=(64 32 18)

cd ..
for skew_index in "${!skews[@]}"
do
    skew="${skews[$skew_index]}"
    for policy in "${policies[@]}"
    do
        if [[ "$policy" == "harmony" || "$policy" == "drop" || "$policy" == "even_split" ]]; then
            batch_size=$batch_size_harmony_drop_even_split
        else
            batch_size=${batch_size_deepspeed_exflow[$skew_index]}
        fi

        python3 src/start_harmony.py \
                --dataset "constant" \
                --num_samples $num_samples \
                --batch_size $batch_size \
                --seq_len $seq_len \
                --model_name "google/switch-base-$num_experts" \
                --num_experts $num_experts \
                --scheduling_policy $policy \
                --expert_cache_size $expert_cache_size \
                --expert_placement "ExFlow/placement/switch-exp${num_experts}-gpu${world_size}.json" \
                --world_size $world_size \
                --eq_tokens $eq_tokens \
                --expert_fetching_strategy "async-cpu" \
                --warmup_rounds $warmup_len \
                --enable_router_skew $enable_skew \
                --enable_router_random $enable_random \
                --enable_router_uniform $enable_uniform \
                --router_skew $skew \
                --router_num_experts_skewed $num_experts_skewed \
                --pa "outputs/exp-switch128-policies-skew/$datetime/$skew-$policy"
    done    
done