num_samples=1000000
output_path="data/policies/dataset/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
world_size=8
num_experts=128


datasets=("random" "wmt19" "bookcorpus" "wikitext")
harmony_batches=(2500 2500 2500 2500)
exflow_batches=(1250 1250 1250 1250)
deepspeed_batches=(1250 1250 1250 1250)
even_split_batches=(2500 2500 2500 2500)
drop_batches=(2500 2500 2500 2500)

cd ..
for i in {0..3}
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

    # python3 src/start_harmony.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${exflow_batches[i]} \
    #     --seq_len $seq_len \
    #     --model_name "google/switch-base-$num_experts" \
    #     --scheduling_policy "exflow" \
    #     --expert_cache_size 16 \
    #     --world_size $world_size \
    #     --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
    #     --pa $output_path/${datasets[i]}/exflow

    # python3 src/start_harmony.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${deepspeed_batches[i]} \
    #     --seq_len $seq_len \
    #     --model_name "google/switch-base-$num_experts" \
    #     --scheduling_policy "deepspeed" \
    #     --expert_cache_size 16 \
    #     --world_size $world_size \
    #     --pa $output_path/${datasets[i]}/deepspeed
    
    # python3 src/start_harmony.py \
    #     --dataset ${datasets[i]} \
    #     --num_samples $num_samples \
    #     --batch_size ${even_split_batches[i]} \
    #     --seq_len $seq_len \
    #     --model_name "google/switch-base-$num_experts" \
    #     --scheduling_policy "even_split" \
    #     --expert_cache_size 16 \
    #     --world_size $world_size \
    #     --pa $output_path/${datasets[i]}/even_split
    
    python3 src/start_harmony.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size ${drop_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "drop" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --pa $output_path/${datasets[i]}/drop
    
done