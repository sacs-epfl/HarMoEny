dataset=constant
num_samples=320000

cd ..
python3 src/start_fastermoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 1000 \
        --seq_len 120 \
        --world_size 8 \
        --num_experts 128 \
        --path "outputs/fastermoe"