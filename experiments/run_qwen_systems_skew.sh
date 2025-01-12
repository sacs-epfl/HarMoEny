datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=12800
seq_len=1024
world_size=6
enable_skew=True
num_experts_skewed=1
warmup_len=3

skews=(0.0 0.5 0.9)
fastmoe_batches=(32 8 4)
deepspeed_batches=(16 4 2)

cd ..
for skew_index in "${!skews[@]}"
do
    skew="${skews[$skew_index]}"
    batch_size=${fastmoe_batches[$skew_index]}
    echo $batch_size
    # python3 src/start_fastmoe.py \
    #     --system_name fastmoe \
    #     --dataset "constant" \
    #     --num_samples $num_samples \
    #     --batch_size $batch_size \
    #     --seq_len $seq_len \
    #     --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    #     --num_experts 60 \
    #     --d_model 2048 \
    #     --type_moe_parent Qwen2MoeDecoderLayer \
    #     --type_moe Qwen2MoeSparseMoeBlock \
    #     --router_tensor_path gate \
    #     --name_experts experts \
    #     --world_size $world_size \
    #     --enable_router_skew $enable_skew \
    #     --router_skew $skew \
    #     --router_num_experts_skewed $num_experts_skewed \
    #     --pa "outputs/exp-qwen-systems-skew/$datetime/$skew-fastmoe"
    
    # python3 src/start_fastmoe.py \
    #     --system_name fastermoe \
    #     --dataset "constant" \
    #     --num_samples $num_samples \
    #     --batch_size $batch_size \
    #     --seq_len $seq_len \
    #     --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    #     --num_experts 60 \
    #     --d_model 2048 \
    #     --type_moe_parent Qwen2MoeDecoderLayer \
    #     --type_moe Qwen2MoeSparseMoeBlock \
    #     --router_tensor_path gate \
    #     --name_experts experts \
    #     --world_size $world_size \
    #     --enable_router_skew $enable_skew \
    #     --router_skew $skew \
    #     --router_num_experts_skewed $num_experts_skewed \
    #     --pa "outputs/exp-qwen-systems-skew/$datetime/$skew-fastermoe"
    
    deepspeed --num_gpus $world_size src/start_deepspeed.py \
        --dataset "constant" \
        --num_samples $num_samples \
        --batch_size ${deepspeed_batches[$skew_index]} \
        --seq_len $seq_len \
        --model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
        --num_experts 60 \
        --d_model 2048 \
        --type_moe_parent Qwen2MoeDecoderLayer \
        --type_moe Qwen2MoeSparseMoeBlock \
        --router_tensor_path gate \
        --name_experts experts \
        --world_size $world_size \
        --enable_router_skew $enable_skew \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/exp-qwen-systems-skew/$datetime/$skew-deepspeed"
done