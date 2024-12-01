num_samples=1000000
output_path="data/systems/skew/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
world_size=8
num_experts=128


datasets=("random" "skew25" "skew50" "skew75" "skew90" "skew95")
harmony_batches=(2500 2500 2500 2500 2500 2500)
exflow_batches=(1250 1100 625 550 450 400)
fastmoe_batches=(1250 1100 625 550 450 400)
fastermoe_batches=(1250 1100 625 550 450 400)
deepspeed_batches=(1250 1100 625 550 450 400)
deepspeed_capacity_factors=(10.0 10.0 25.0 30.0 35.0 37.5)

cd ..
for i in {0..5}
do
    # python3 src/start_harmony.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${harmony_batches[i]} \
    #     --seq_len $seq_len \
    #     --model_name "google/switch-base-$num_experts" \
    #     --scheduling_policy "harmony" \
    #     --expert_cache_size 16 \
    #     --world_size $world_size \
    #     --pa $output_path/${datasets[i]}/harmony

    python3 src/start_harmony.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size ${exflow_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "exflow" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
        --pa $output_path/${datasets[i]}/exflow

    # python3 src/start_fastmoe.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${fastmoe_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size $world_size \
    #     --pa $output_path/${datasets[i]}/fastmoe

    # python3 src/start_fastermoe.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${fastermoe_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size $world_size \
    #     --pa $output_path/${datasets[i]}/fastermoe

    # deepspeed --num_gpus $world_size src/start_deepspeed.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${deepspeed_batches[i]} \
    #     --seq_len $seq_len \
    #     --num_experts $num_experts \
    #     --world_size $world_size \
    #     --capacity_factor ${deepspeed_capacity_factors[i]} \
    #     --pa $output_path/${datasets[i]}/deepspeed
done

