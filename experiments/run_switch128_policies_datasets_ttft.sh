datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=15360
seq_len=1024
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
eq_tokens=1512
warmup_len=3

policies=("deepspeed" "harmony" "drop" "even_split" "exflow")
datasets=("wmt19" "bookcorpus" "wikitext" "random")
batch_size=64

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
                --model_name "google/switch-base-$num_experts" \
                --num_experts $num_experts \
                --scheduling_policy $policy \
                --expert_cache_size $expert_cache_size \
                --expert_placement "ExFlow/placement/switch-exp${num_experts}-gpu${world_size}.json" \
                --world_size $world_size \
                --eq_tokens $eq_tokens \
                --expert_fetching_strategy "async-cpu" \
                --warmup_rounds $warmup_len \
                --pa "outputs/exp-switch128-policies-dataset-ttft/$datetime/$dataset-$policy"
    done
done
