dataset=bookcorpus
num_samples=160000

cd ..
deepspeed --num_gpus 8 src/start_deepspeed.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 1000 \
        --seq_len 60 \
        --num_experts 64 \
        --world_size 8 \
        --path "outputs/deepspeed"