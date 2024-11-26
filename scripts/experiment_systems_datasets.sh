num_samples=800000
output_path="data/systems/dataset/$(date +"%Y-%m-%d_%H-%M")"

cd ..
########## RANDOM ############
# python3 src/start_harmony.py \
#     --dataset random \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --model_name "google/switch-base-128" \
#     --scheduling_policy "harmony" \
#     --expert_cache_size 16 \
#     --world_size 8 \
#     --pa "$output_path/random/harmony"

python3 src/start_harmony.py \
    --dataset random \
    --num_samples $num_samples \
    --batch_size 2250 \
    --seq_len 60 \
    --model_name "google/switch-base-128" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size 8 \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/random/exflow"

# python3 src/start_fastmoe.py \
#     --dataset random \
#     --num_samples $num_samples \
#     --batch_size 2500 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/random/fastmoe"

# python3 src/start_fastermoe.py \
#     --dataset random \
#     --num_samples $num_samples \
#     --batch_size 2500 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/random/fastermoe"

# deepspeed --num_gpus 8 src/start_deepspeed.py \
#     --dataset random \
#     --num_samples $num_samples \
#     --batch_size 4500 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/random/deepspeed"

# ######## WIKITEXT ##############
# python3 src/start_harmony.py \
#     --dataset wikitext \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --model_name "google/switch-base-128" \
#     --scheduling_policy "harmony" \
#     --expert_cache_size 16 \
#     --world_size 8 \
#     --pa "$output_path/wikitext/harmony"

python3 src/start_harmony.py \
    --dataset wikitext \
    --num_samples $num_samples \
    --batch_size 2250 \
    --seq_len 60 \
    --model_name "google/switch-base-128" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size 8 \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/wikitext/exflow"

# python3 src/start_fastmoe.py \
#     --dataset wikitext \
#     --num_samples $num_samples \
#     --batch_size 4000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/wikitext/fastmoe"

# python3 src/start_fastermoe.py \
#     --dataset wikitext \
#     --num_samples $num_samples \
#     --batch_size 4000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/wikitext/fastermoe"

# deepspeed --num_gpus 8 src/start_deepspeed.py \
#     --dataset wikitext \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/wikitext/deepspeed"

# ######## BOOKCORPUS ##############
# python3 src/start_harmony.py \
#     --dataset bookcorpus \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --model_name "google/switch-base-128" \
#     --scheduling_policy "harmony" \
#     --expert_cache_size 16 \
#     --world_size 8 \
#     --pa "$output_path/bookcorpus/harmony"

python3 src/start_harmony.py \
    --dataset bookcorpus \
    --num_samples $num_samples \
    --batch_size 2250 \
    --seq_len 60 \
    --model_name "google/switch-base-128" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size 8 \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/bookcorpus/exflow"

# python3 src/start_fastmoe.py \
#     --dataset bookcorpus \
#     --num_samples $num_samples \
#     --batch_size 4000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/bookcorpus/fastmoe"

# python3 src/start_fastermoe.py \
#     --dataset bookcorpus \
#     --num_samples $num_samples \
#     --batch_size 4000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/bookcorpus/fastermoe"

# deepspeed --num_gpus 8 src/start_deepspeed.py \
#     --dataset bookcorpus \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/bookcorpus/deepspeed"

# ######## WMT19 ##############
# python3 src/start_harmony.py \
#     --dataset wmt19 \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --model_name "google/switch-base-128" \
#     --scheduling_policy "harmony" \
#     --expert_cache_size 16 \
#     --world_size 8 \
#     --pa "$output_path/wmt19/harmony"

python3 src/start_harmony.py \
    --dataset wmt19 \
    --num_samples $num_samples \
    --batch_size 2250 \
    --seq_len 60 \
    --model_name "google/switch-base-128" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size 8 \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/wmt19/exflow"

# python3 src/start_fastmoe.py \
#     --dataset wmt19 \
#     --num_samples $num_samples \
#     --batch_size 4000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/wmt19/fastmoe"

# python3 src/start_fastermoe.py \
#     --dataset wmt19 \
#     --num_samples $num_samples \
#     --batch_size 4000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/wmt19/fastermoe"

# deepspeed --num_gpus 8 src/start_deepspeed.py \
#     --dataset wmt19 \
#     --num_samples $num_samples \
#     --batch_size 5000 \
#     --seq_len 60 \
#     --num_experts 128 \
#     --world_size 8 \
#     --pa "$output_path/wmt19/deepspeed"
