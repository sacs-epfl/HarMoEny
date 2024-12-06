dataset="wikitext"
num_samples=1000000
output_path="data/systems/experts/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
world_size=8

num_experts=(8 16 32 64 128)

harmony_batches=(4000 2000 2500 2500)
exflow_batches=(625 1000 1250 1250)
fastmoe_batches=(1000 2000 2250 2250)
fastermoe_batches=(1000 2000 2250 2250)
deepspeed_batches=(250 800 1000 1000)
deepspeed_capacity_factors=(15.0 15.0 15.0)

cd ..
for i in {0..3}
do
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${harmony_batches[i]} \
        --seq_len $seq_len \
        --model_name google/switch-base-${num_experts[i]} \
        --scheduling_policy "harmony" \
        --expert_cache_size $((${num_experts[i]} / $world_size)) \
        --world_size $world_size \
        --pa $output_path/${num_experts[i]}/harmony

    # python3 src/start_harmony.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${exflow_batches[i]} \
    #     --seq_len $seq_len \
    #     --model_name google/switch-base-${num_experts[i]} \
    #     --scheduling_policy "exflow" \
    #     --expert_cache_size $((${num_experts[i]} / $world_size)) \
    #     --world_size $world_size \
    #     --expert_placement ExFlow/placement/exp${num_experts}_gpu$world_size.json \
    #     --pa $output_path/${num_experts[i]}/exflow

    # python3 src/start_fastmoe.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${fastmoe_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts ${num_experts[i]} \
    #     --world_size $world_size \
    #     --pa $output_path/${num_experts[i]}/fastmoe

    # python3 src/start_fastermoe.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${fastermoe_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts ${num_experts[i]} \
    #     --world_size $world_size \
    #     --pa $output_path/${num_experts[i]}/fastermoe

    # deepspeed --num_gpus $world_size src/start_deepspeed.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${deepspeed_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts ${num_experts[i]} \
    #     --world_size $world_size \
    #     --capacity_factor ${deepspeed_capacity_factors[i]} \
    #     --pa $output_path/${num_experts[i]}/deepspeed

done