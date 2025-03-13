datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=2880
seq_len=1024
world_size=6
expert_cache_size=10
expert_fetching_strategy="async-cpu"
eq_tokens=1024 # will prob need updating
warmup_len=3

policies=("deepspeed" "harmony" "even_split" "drop" "exflow")
datasets=("wmt19" "bookcorpus" "wikitext" "random")

batch_size=16

cd ..
for dataset_index in "${!datasets[@]}"
do
    dataset="${datasets[$dataset_index]}"
    for policy in "${policies[@]}"
    do
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
                --scheduling_policy $policy \
                --expert_cache_size $expert_cache_size \
                --expert_placement "ExFlow/placement/qwen-gpu$world_size.json" \
                --world_size $world_size \
                --eq_tokens $eq_tokens \
                --expert_fetching_strategy "async-cpu" \
                --warmup_rounds $warmup_len \
                --pa "outputs/exp-qwen-policies-dataset-ttft/$datetime/$dataset-$policy"
    done    
done
