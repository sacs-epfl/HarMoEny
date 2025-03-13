datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=11520
seq_len=1024
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
eq_tokens=1512
warmup_len=3

datasets=("wmt19" "bookcorpus" "wikitext" "random")
batch_size=48

cd ..
for dataset_index in "${!datasets[@]}"
do
    dataset="${datasets[$dataset_index]}"

    python3 src/start_harmony.py \
        --dataset $dataset \
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
        --pa "outputs/exp-switch128-systems-dataset-ttft/$datetime/$dataset-harmony"

    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --scheduling_policy exflow \
        --expert_cache_size $expert_cache_size \
        --expert_placement "ExFlow/placement/switch-exp${num_experts}-gpu${world_size}.json" \
        --world_size $world_size \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --pa "outputs/exp-switch128-systems-dataset-ttft/$datetime/$dataset-exflow"

    python3 src/start_fastmoe.py \
        --system_name fastmoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --pa "outputs/exp-switch128-systems-dataset-ttft/$datetime/$dataset-fastmoe"

    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --warmup_rounds $warmup_len \
        --pa "outputs/exp-switch128-systems-dataset-ttft/$datetime/$dataset-fastermoe"
done
