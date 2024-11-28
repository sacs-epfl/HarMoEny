dataset=bookcorpus
num_samples=200000
num_experts=128
num_gpus=2
policy="harmony"
seq_len=120

cd ..
# python3 src/start_harmony.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --seq_len $seq_len \
#         --model_name "google/switch-base-$num_experts" \
#         --scheduling_policy $policy \
#         --expert_cache_size $(($num_experts / $num_gpus)) \
#         --world_size $num_gpus \
#         --path "outputs/harmony"

python3 src/start_harmony.py \
        --dataset bookcorpus \
        --num_samples 80000 \
        --seq_len 120 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "harmony" \
        --expert_cache_size 16 \
        --world_size 8 \
        --batch_size 2000 \
        --path "outputs/harmony/run" 

