num_samples=25600
output_path="data/systems/dataset/$(date +"%Y-%m-%d_%H-%M")"
seq_len=512
batch_size=64
world_size=8
num_experts=128

datasets=("random" "bookcorpus" "wikitext" "wmt19")

cd ..
for i in {0..0}
do
    python3 src/start_harmony.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --pa $output_path/${datasets[i]}/harmony

    python3 src/start_harmony.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "exflow" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
        --pa $output_path/${datasets[i]}/exflow

    python3 src/start_fastmoe.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --num_experts $num_experts \
        --world_size $world_size \
        --pa $output_path/${datasets[i]}/fastmoe

    python3 src/start_fastermoe.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --num_experts $num_experts \
        --world_size $world_size \
        --pa $output_path/${datasets[i]}/fastermoe

    deepspeed --num_gpus $world_size src/start_deepspeed.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --num_experts $num_experts \
        --world_size $world_size \
        --capacity_factor 30.0 \
        --pa $output_path/${datasets[i]}/deepspeed

done