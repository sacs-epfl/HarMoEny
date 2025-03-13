datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=3600
seq_len=1024
world_size=6
expert_cache_size=10
expert_fetching_strategy="async-cpu"
eq_tokens=1024 # will prob need updating
warmup_len=3

datasets=("wmt19" "bookcorpus" "wikitext" "random")
batch_size=20

cd ..
for dataset_index in "${!datasets[@]}"
do
    dataset="${datasets[$dataset_index]}"

    python3 src/start_harmony.py \
        --dataset $dataset \
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
        --scheduling_policy harmony \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --pa "outputs/exp-qwen-systems-dataset-ttft/$datetime/$dataset-harmony"

    python3 src/start_harmony.py \
        --dataset $dataset \
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
        --scheduling_policy exflow \
        --expert_cache_size $expert_cache_size \
        --expert_placement "ExFlow/placement/qwen-gpu$world_size.json" \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --pa "outputs/exp-qwen-systems-dataset-ttft/$datetime/$dataset-exflow"

    python3 src/start_fastmoe.py \
        --system_name fastmoe \
        --dataset $dataset \
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
        --pa "outputs/exp-qwen-systems-dataset-ttft/$datetime/$dataset-fastmoe"

    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset $dataset \
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
        --pa "outputs/exp-qwen-systems-dataset-ttft/$datetime/$dataset-fastermoe"
done
