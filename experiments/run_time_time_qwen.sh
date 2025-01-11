datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=3840
seq_len=1024
batch_size=48
world_size=6
expert_cache_size=10
enable_skew=True
skew=90
num_experts_skewed=10
eq_tokens=1024
warmup_len=3

cd ..
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
        --scheduling_policy "deepspeed" \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/timetime-qwen/$datetime/harmony-no-sched"

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
        --scheduling_policy "harmony" \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "sync-cpu" \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/timetime-qwen/$datetime/harmony-sync"

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
        --scheduling_policy "harmony" \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/timetime-qwen/$datetime/harmony-async"


