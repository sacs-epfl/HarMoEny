num_samples=800000
output_path="data/systems/experts/$(date +"%Y-%m-%d_%H-%M")"
experts=(8 16 32 64 128)
dataset=wikitext
seq_len=120
num_gpus=8

cd ..
for e in ${experts[@]}; do 
    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --model_name "google/switch-base-$e" \
        --scheduling_policy "harmony" \
        --expert_cache_size $(($e / $num_gpus)) \
        --world_size $num_gpus \
        --pa "$output_path/$e/harmony"

    python3 src/start_harmony.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --model_name "google/switch-base-$e" \
        --scheduling_policy "exflow" \
        --expert_cache_size $(($e / $num_gpus)) \
        --world_size $num_gpus \
        --expert_placement "ExFlow/placement/exp${e}_gpu${num_gpus}.json" \
        --pa "$output_path/$e/exflow"

    python3 src/start_fastmoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --num_experts $e \
        --world_size $num_gpus \
        --pa "$output_path/$e/fastmoe"

    python3 src/start_fastermoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --num_experts $e \
        --world_size $num_gpus \
        --pa "$output_path/$e/fastermoe"

    deepspeed --num_gpus $num_gpus src/start_deepspeed.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --seq_len $seq_len \
        --num_experts $e \
        --world_size $num_gpus \
        --pa "$output_path/$e/deepspeed"
done 

