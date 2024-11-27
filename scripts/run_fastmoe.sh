dataset=bookcorpus
num_samples=320000

cd ..
python3 src/start_fastmoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 2000 \
        --seq_len 120 \
        --world_size 8 \
        --num_experts 128 \
        --path "outputs/fastmoe"