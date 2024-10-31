num_samples=1000000
output_path="outputs/experiment_3/$(date +"%Y-%m-%d_%H-%M")"

datasets=("random" "wikitext" "bookcorpus" "wmt19")
cd ..
for dataset in "${datasets[@]}" 
do 
    python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5250 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "adnexus" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/$dataset/harmonymoe"

    python3 src/start.py \
        --system_name fastmoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 2500 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --world_size 8 \
        --pa "$output_path/$dataset/fastmoe"

    python3 src/start.py \
        --system_name fastermoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 2500 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --world_size 8 \
        --pa "$output_path/$dataset/fastermoe"
    
    deepspeed --num_gpus 8 src/start.py \
        --system_name deepspeed-inference \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 500 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --pa "$output_path/$dataset/deepspeed-inference"
done