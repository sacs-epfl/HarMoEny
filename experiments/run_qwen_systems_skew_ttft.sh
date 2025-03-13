datetime=$(date +"%Y-%m-%d_%H-%M")

enable_skew=True
num_experts_skewed=1
seq_len=1024
world_size=6
expert_cache_size=10
expert_fetching_strategy="async-cpu"
eq_tokens=1512 # will prob need updating
warmup_len=3

skews=(0.9 0.5 0.0)
batches=(4 8 32)
num_sampless=(720 1440 5760)

cd ..
for skew_index in "${!skews[@]}"
do
    skew="${skews[$skew_index]}"
    batch_size="${batches[$skew_index]}"
    num_samples="${num_sampless[$skew_index]}"

    python3 src/start_harmony.py \
        --dataset "constant" \
        --num_samples $num_sample \
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
        --scheduling_policy harmony \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-qwen-systems-skew-ttft/$datetime/$skew-harmony"

    python3 src/start_harmony.py \
        --dataset "constant" \
        --num_samples $num_sample \
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
        --scheduling_policy exflow \
        --expert_cache_size $expert_cache_size \
        --expert_placement "ExFlow/placement/qwen-gpu$world_size.json" \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-qwen-systems-skew-ttft/$datetime/$skew-exflow"

    python3 src/start_fastmoe.py \
        --system_name fastmoe \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        --num_experts 60 \
        --d_model 2048 \
        --type_moe_parent Qwen2MoeDecoderLayer \
        --type_moe Qwen2MoeSparseMoeBlock \
        --router_tensor_path gate \
        --name_experts experts \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-qwen-systems-skew-ttft/$datetime/$skew-fastmoe"

    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        --num_experts 60 \
        --d_model 2048 \
        --type_moe_parent Qwen2MoeDecoderLayer \
        --type_moe Qwen2MoeSparseMoeBlock \
        --router_tensor_path gate \
        --name_experts experts \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-qwen-systems-skew-ttft/$datetime/$skew-fastermoe"
done
