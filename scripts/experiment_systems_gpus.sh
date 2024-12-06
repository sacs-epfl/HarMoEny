dataset="wikitext"
num_samples=1000000
output_path="data/systems/gpus/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
num_experts=128

num_gpus=(2 4 8)

harmony_batches=(1250 2000 2500)
exflow_batches=(625 1000 1250)
fastmoe_batches=(1000 2000 2250)
fastermoe_batches=(1000 2000 2250)
deepspeed_batches=(250 800 1000)
deepspeed_capacity_factors=(15.0 15.0 15.0)

cd ..
for i in {0..0}
do
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${harmony_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size $(($num_experts / ${num_gpus[i]})) \
        --world_size ${num_gpus[i]} \
        --eq_tokens 1024 \
        --pa $output_path/${num_gpus[i]}/harmony

    # python3 src/start_harmony.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${exflow_batches[i]} \
    #     --seq_len $seq_len \
    #     --model_name "google/switch-base-$num_experts" \
    #     --scheduling_policy "exflow" \
    #     --expert_cache_size $(($num_experts / ${num_gpus[i]})) \
    #     --world_size ${num_gpus[i]} \
    #     --expert_placement ExFlow/placement/exp${num_experts}_gpu${num_gpus[i]}.json \
    #     --pa $output_path/${num_gpus[i]}/exflow

    # python3 src/start_fastmoe.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${fastmoe_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size ${num_gpus[i]} \
    #     --pa $output_path/${num_gpus[i]}/fastmoe

    # python3 src/start_fastermoe.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${fastermoe_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size ${num_gpus[i]} \
    #     --pa $output_path/${num_gpus[i]}/fastermoe

    # deepspeed --num_gpus ${num_gpus[i]} src/start_deepspeed.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${deepspeed_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size ${num_gpus[i]} \
    #     --capacity_factor ${deepspeed_capacity_factors[i]} \
    #     --pa $output_path/${num_gpus[i]}/deepspeed

done
