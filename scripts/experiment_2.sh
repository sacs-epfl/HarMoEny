num_samples=1000000
output_path="outputs/experiment_2/$(date +"%Y-%m-%d_%H-%M")"

datasets=("random" "wikitext" "bookcorpus" "wmt19")
cd ..
for dataset in "${datasets[@]}" 
do 
    python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 2000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "deepspeed" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/$dataset/deepspeed"

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
        --pa "$output_path/$dataset/adnexus"
    
    python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5250 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "drop" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/$dataset/drop"
    
    python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5250 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "even_split" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/$dataset/even_split"
    
    python3 src/start.py \
        --system_name harmony \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 5000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --scheduling_policy "adfabricus" \
        --expert_cache_size 16 \
        --world_size 8 \
        --pa "$output_path/$dataset/adfabricus"
done
