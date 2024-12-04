dataset="constant"
num_samples=1000000
num_experts=128
world_size=8
seq_len=120
output_path="data/systems/router_skew/$(date +"%Y-%m-%d_%H-%M")"

skews=(0.0 0.5 0.95)
harmony_batches=(2500 2500 2500)
exflow_batches=(2500 1250 400)
fastmoe_batches=(2500 1250 400)
fastermoe_batches=(2500 1800 1000)
deepspeed_batches=(2500 250 100)
deepspeed_capacity_factors=(1.0 64.0 121.6) # skew * num_experts. Except 0 means uniform.

cd ..
for i in {0..2}
do 
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${harmony_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --enable_router_skew True \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/harmony

    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${exflow_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "exflow" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
        --enable_router_skew True \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/exflow

    python3 src/start_fastmoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${fastmoe_batches[i]} \
        --seq_len $seq_len \
        --num_experts $num_experts \
        --world_size $world_size \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/fastmoe

    python3 src/start_fastermoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${fastermoe_batches[i]} \
        --seq_len $seq_len \
        --num_experts $num_experts \
        --world_size $world_size \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/fastermoe

    # deepspeed --num_gpus $world_size src/start_deepspeed.py \
    #     --dataset $dataset \
    #     --num_samples $num_samples \
    #     --batch_size ${deepspeed_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size $world_size \
    #     --capacity_factor ${deepspeed_capacity_factors[i]} \
    #     --router_skew ${skews[i]} \
    #     --pa $output_path/${skews[i]}/deepspeed
done

# Deepspeed takes too long so merely take predictions, have batch time printed for deepspeed and take the average of a few rounds