datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=640
seq_len=1024
world_size=4
expert_cache_size=2
expert_fetching_strategy="async-cpu"
eq_tokens=1512 # 2048
warmup_len=3

enable_skew=True
num_experts_skewed=1

# TODO run exflow thing
# exflow will need to be rerun for this model exflow removed temporarily
policies=("even_split" "harmony" "deepspeed" "drop")
skews=(0.0 0.5 0.9)

batch_size_deepspeed_exflow=(16 8 2)
batch_size_harmony_drop_even_split=16

# OVERRIDES 
skews=(0.9)
batch_size_deepspeed_exflow=(2)
batch_size_harmony_drop_even_split=16

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
                --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
                --loader transformers \
                --d_model 4096 \
                --model_dtype int8 \
                --num_experts 8 \
                --type_moe_parent MixtralDecoderLayer \
                --type_moe MixtralSparseMoeBlock \
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
                --router_skew $skew \
                --router_num_experts_skewed $num_experts_skewed \
                --pa "outputs/exp-mixtral-int8-policies-skew/$datetime/$skew-$policy"
    done    
done