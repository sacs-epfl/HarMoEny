num_samples=1000000
output_path="data/systems/datasets/$(date +"%Y-%m-%d_%H-%M")"
seq_len=120
world_size=8
num_experts=128

cd ..
########## COCKTAIL ############
python3 src/start_harmony.py \
    --dataset cocktail \
    --num_samples $num_samples \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --pa "$output_path/cocktail/harmony"

python3 src/start_harmony.py \
    --dataset cocktail \
    --num_samples $num_samples \
    --batch_size 1400 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
    --pa "$output_path/cocktail/exflow"

python3 src/start_fastmoe.py \
    --dataset cocktail \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/cocktail/fastmoe"

python3 src/start_fastermoe.py \
    --dataset cocktail \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/cocktail/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset cocktail \
    --num_samples $num_samples \
    --batch_size 1300 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/cocktail/deepspeed"

########## SKEW50 ############
python3 src/start_harmony.py \
    --dataset skew50 \
    --num_samples $num_samples \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --pa "$output_path/skew50/harmony"

python3 src/start_harmony.py \
    --dataset skew50 \
    --num_samples $num_samples \
    --batch_size 700 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
    --pa "$output_path/skew50/exflow"

python3 src/start_fastmoe.py \
    --dataset skew50 \
    --num_samples $num_samples \
    --batch_size 625 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/skew50/fastmoe"

python3 src/start_fastermoe.py \
    --dataset skew50 \
    --num_samples $num_samples \
    --batch_size 625 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/skew50/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset skew50 \
    --num_samples $num_samples \
    --batch_size 650 \
    --seq_len $seq_len \
    --capacity_factor 25.0 \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/skew50/deepspeed"

########## RANDOM ############
python3 src/start_harmony.py \
    --dataset random \
    --num_samples $num_samples \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --pa "$output_path/random/harmony"

python3 src/start_harmony.py \
    --dataset random \
    --num_samples $num_samples \
    --batch_size 1400 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --expert_placement "ExFlow/placement/exp${num_experts}_gpu${world_size}.json" \
    --pa "$output_path/random/exflow"

python3 src/start_fastmoe.py \
    --dataset random \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/random/fastmoe"

python3 src/start_fastermoe.py \
    --dataset random \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/random/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset random \
    --num_samples $num_samples \
    --batch_size 1300 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/random/deepspeed"

######## WIKITEXT ##############
python3 src/start_harmony.py \
    --dataset wikitext \
    --num_samples $num_samples \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --pa "$output_path/wikitext/harmony"

python3 src/start_harmony.py \
    --dataset wikitext \
    --num_samples $num_samples \
    --batch_size 1600 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/wikitext/exflow"

python3 src/start_fastmoe.py \
    --dataset wikitext \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/wikitext/fastmoe"

python3 src/start_fastermoe.py \
    --dataset wikitext \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/wikitext/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset wikitext \
    --num_samples $num_samples \
    --batch_size 1300 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/wikitext/deepspeed"

######## BOOKCORPUS ##############
python3 src/start_harmony.py \
    --dataset bookcorpus \
    --num_samples $num_samples \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --pa "$output_path/bookcorpus/harmony"

python3 src/start_harmony.py \
    --dataset bookcorpus \
    --num_samples $num_samples \
    --batch_size 1400 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/bookcorpus/exflow"

python3 src/start_fastmoe.py \
    --dataset bookcorpus \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/bookcorpus/fastmoe"

python3 src/start_fastermoe.py \
    --dataset bookcorpus \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/bookcorpus/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset bookcorpus \
    --num_samples $num_samples \
    --batch_size 1300 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/bookcorpus/deepspeed"

######## WMT19 ##############
python3 src/start_harmony.py \
    --dataset wmt19 \
    --num_samples $num_samples \
    --batch_size 2500 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "harmony" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --pa "$output_path/wmt19/harmony"

python3 src/start_harmony.py \
    --dataset wmt19 \
    --num_samples $num_samples \
    --batch_size 1400 \
    --seq_len $seq_len \
    --model_name "google/switch-base-$num_experts" \
    --scheduling_policy "exflow" \
    --expert_cache_size 16 \
    --world_size $world_size \
    --expert_placement "ExFlow/placement/exp128_gpu8.json" \
    --pa "$output_path/wmt19/exflow"

python3 src/start_fastmoe.py \
    --dataset wmt19 \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/wmt19/fastmoe"

python3 src/start_fastermoe.py \
    --dataset wmt19 \
    --num_samples $num_samples \
    --batch_size 1250 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/wmt19/fastermoe"

deepspeed --num_gpus $world_size src/start_deepspeed.py \
    --dataset wmt19 \
    --num_samples $num_samples \
    --batch_size 1300 \
    --seq_len $seq_len \
    --num_experts $num_experts \
    --world_size $world_size \
    --pa "$output_path/wmt19/deepspeed"
