datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=12800
seq_len=1024
world_size=6
warmup_len=3

datasets=("wmt19" "bookcorpus" "wikitext" "random")
fastmoe_batch=20

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
        --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        --num_experts 60 \
        --d_model 2048 \
        --type_moe_parent Qwen2MoeDecoderLayer \
        --type_moe Qwen2MoeSparseMoeBlock \
        --router_tensor_path gate \
        --name_experts experts \
        --world_size $world_size \
        --pa "outputs/exp-qwen-systems-dataset/$datetime/$dataset-fastmoe"
    
    python3 src/start_fastmoe.py \
        --system_name fastermoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $fastmoe_batch \
        --seq_len $seq_len \
        --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        --num_experts 60 \
        --d_model 2048 \
        --type_moe_parent Qwen2MoeDecoderLayer \
        --type_moe Qwen2MoeSparseMoeBlock \
        --router_tensor_path gate \
        --name_experts experts \
        --world_size $world_size \
        --pa "outputs/exp-qwen-systems-dataset/$datetime/$dataset-fastermoe"
done