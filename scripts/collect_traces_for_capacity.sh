num_experts=128
num_gpus=8
policy="harmony"
seq_len=120
num_iters=5

batch_sizes=(1200 300 650 1200 1300 1300 1300) 
datasets=("cocktail" "skew95" "skew50" "random" "wikitext" "bookcorpus" "wmt19")

cd ..
for i in {0..6}
do
    python3 src/start_harmony.py \
        --dataset ${datasets[i]} \
        --num_samples $(($num_gpus * $num_iters * ${batch_sizes[i]})) \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy deepspeed \
        --expert_cache_size $(($num_experts / $num_gpus)) \
        --world_size $num_gpus \
        --batch_size ${batch_sizes[i]} \
        --path outputs/capacity/${datasets[i]}
done