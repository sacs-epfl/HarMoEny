num_samples=1000000
output_path="data/systems/experts/$(date +"%Y-%m-%d_%H-%M")"
e=128
dataset=random
seq_len=120
num_gpus=8
batch_size=1200

cd .. 
python3 src/start_harmony.py \
    --dataset $dataset \
    --num_samples $num_samples \
    --seq_len $seq_len \
    --model_name "google/switch-base-$e" \
    --scheduling_policy "exflow" \
    --expert_cache_size $(($e / $num_gpus)) \
    --world_size $num_gpus \
    --expert_placement "ExFlow/placement/exp${e}_gpu${num_gpus}.json" \
    --batch_size $batch_size \
    --pa "$output_path/$e/exflow"