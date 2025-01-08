datetime=$(date +"%Y-%m-%d_%H-%M")

cd ..
python3 src/start_harmony.py \
    --dataset bookcorpus \
    --num_samples 48 \
    --batch_size 8 \
    --seq_len 1024 \
    --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    --loader transformers \
    --d_model 2048 \
    --num_experts 60 \
    --type_moe_parent Qwen2MoeDecoderLayer \
    --type_moe Qwen2MoeSparseMoeBlock \
    --router_tensor_path gate \
    --name_experts experts \
    --scheduling_policy deepspeed \
    --expert_cache_size 10 \
    --world_size 6 \
    --expert_fetching_strategy "async-cpu" \
    --warmup_rounds 0 \
    --pa "outputs/qwen/$datetime"