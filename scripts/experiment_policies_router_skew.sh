num_samples=1000000
output_path="data/policies/router_skew/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
world_size=8
num_experts=128
dataset="constant"

skews=(0.0 0.5 0.95)
harmony_batches=(2500 2500 2500)
exflow_batches=(2000 1000 625)
deepspeed_batches=(2000 1000 625) 
even_split_batches=(2500 2500 2500)
drop_batches=(2500 2500 2500 2500)

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

    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${deepspeed_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --enable_router_skew True \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/deepspeed
    
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${even_split_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "even_split" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --enable_router_skew True \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/even_split
    
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size ${drop_batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "drop" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --enable_router_skew True \
        --router_skew ${skews[i]} \
        --pa $output_path/${skews[i]}/drop
    
done