dataset=bookcorpus
num_samples=320000

cd ..
python3 src/start_fastermoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 4000 \
        --seq_len 60 \
        --world_size 8 \
        --num_experts 64 \
        --path "outputs/fastermoe"