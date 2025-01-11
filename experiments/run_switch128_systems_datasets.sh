datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=12800
seq_len=1024
world_size=8
num_experts=128
warmup_len=3

datasets=("wmt19" "bookcorpus" "wikitext" "random")
fastmoe_batch=48

cd ..
for dataset_index in "${!datasets[@]}"
do
    dataset="${datasets[$dataset_index]}"
    python3 src/start_fastmoe.py \
        --system_name fastmoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $fastmoe_batch \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --pa "outputs/exp-switch128-systems-dataset/$datetime/$dataset-fastmoe"
    
    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $fastmoe_batch \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --world_size $world_size \
        --pa "outputs/exp-switch128-systems-dataset/$datetime/$dataset-fastermoe"
done