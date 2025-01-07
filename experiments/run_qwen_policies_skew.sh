datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=1920
seq_len=1024
world_size=6
expert_cache_size=10
expert_fetching_strategy="async-cpu"
eq_tokens=1024 # will prob need updating
warmup_len=3

enable_skew=True
num_experts_skewed=1
enable_random=False
enable_uniform=False

# exflow will need to be rerun for this model exflow removed temporarily
policies=("deepspeed" "harmony" "even_split" "drop")
skews=(0.0 0.5 0.9)


batch_size_deepspeed_exflow=(32 8 4)
batch_size_harmony_drop_even_split=32

skews=(0.5 0.9)
batch_size_deepspeed_exflow=(8 4)

cd ..
for skew_index in "${!skews[@]}"
do
    skew="${skews[$skew_index]}"
    for policy in "${policies[@]}"
    do
        if [[ "$policy" == "harmony" || "$policy" == "drop" || "$policy" == "even_split" ]]; then
            batch_size=$batch_size_harmony_drop_even_split
        else
            batch_size=$batch_size_deepspeed_exflow
        fi

        python3 src/start_harmony.py \
                --dataset "constant" \
                --num_samples $num_samples \
                --batch_size $batch_size \
                --seq_len $seq_len \
                --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
                --loader transformers \
                --d_model 2048 \
                --num_experts 60 \
                --type_moe_parent Qwen2MoeDecoderLayer \
                --type_moe Qwen2MoeSparseMoeBlock \
                --router_tensor_path gate \
                --name_experts experts \
                --scheduling_policy $policy \
                --expert_cache_size $expert_cache_size \
                --expert_placement "ExFlow/placement/8_gpu${world_size}.json" \
                --world_size $world_size \
                --eq_tokens $eq_tokens \
                --expert_fetching_strategy "async-cpu" \
                --warmup_rounds $warmup_len \
                --enable_router_skew $enable_skew \
                --enable_router_random $enable_random \
                --enable_router_uniform $enable_uniform \
                --router_skew $skew \
                --router_num_experts_skewed $num_experts_skewed \
                --pa "outputs/exp-qwen-policies-skew/$datetime/$skew-$policy"
    done    
done