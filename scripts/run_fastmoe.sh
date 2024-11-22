dataset=bookcorpus
num_samples=4000

cd ..
python3 src/start_fastmoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 100 \
        --seq_len 60 \
        --world_size 8 \
        --num_experts 64 \
        --path "outputs/fastmoe"