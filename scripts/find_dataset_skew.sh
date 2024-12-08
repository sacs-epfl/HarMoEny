num_samples=100000
output_path="data/stats/dataset_skew/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
world_size=8
num_experts=128

datasets=("random" "bookcorpus" "wikitext" "wmt19")
batches=(1250 1250 1250 1250)

cd ..
for i in {0..3}
do
    python3 src/start_harmony.py \
        --dataset ${datasets[i]} \
        --num_samples $num_samples \
        --batch_size ${batches[i]} \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size $world_size \
        --pa $output_path/${datasets[i]}
done 


