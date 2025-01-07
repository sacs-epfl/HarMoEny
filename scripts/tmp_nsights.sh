datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=256
seq_len=1024
world_size=4
expert_cache_size=2
expert_fetching_strategy="async-cpu"
eq_tokens=1024 # will prob need updating
warmup_len=0

# exflow will need to be rerun for this model exflow removed temporarily
policies=("even_split")
datasets=("random")

batch_size_deepspeed_exflow=32
batch_size_harmony_drop_even_split=64

cd ..
for dataset_index in "${!datasets[@]}"
do
    dataset="${datasets[$dataset_index]}"
    for policy in "${policies[@]}"
    do
        if [[ "$policy" == "harmony" || "$policy" == "drop" || "$policy" == "even_split" ]]; then
            batch_size=$batch_size_harmony_drop_even_split
        else
            batch_size=$batch_size_deepspeed_exflow
        fi

        nsys profile --trace=cuda,osrt,nvtx --force-overwrite=true --output="$dataset-$policy" python3 src/start_harmony.py \
                --dataset $dataset \
                --num_samples $num_samples \
                --batch_size $batch_size \
                --seq_len $seq_len \
                --model_name "hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4" \
                --loader awq \
                --d_model 4096 \
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
                --pa "outputs/garbage/$datetime/$dataset-$policy"
    done    
done