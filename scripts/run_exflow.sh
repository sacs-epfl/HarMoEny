num_samples=800000
output_path="data/systems/experts/$(date +"%Y-%m-%d_%H-%M")"
e=128
dataset=wikitext
seq_len=120
num_gpus=8

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
    --start_batch_size 2380 \
    --pa "$output_path/$e/exflow"