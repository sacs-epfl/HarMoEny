dataset=wmt19
num_samples=450000
output_path="outputs/2024-10-25"

cd ..
python3 start.py \
        --system_name fastermoe \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 3000 \
        --seq_len 60 \
        --model_name "google/switch-base-128" \
        --world_size 8 \
        --pa "$output_path/$dataset"