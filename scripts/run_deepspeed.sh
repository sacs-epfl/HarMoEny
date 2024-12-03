dataset=constant
num_samples=160000

cd ..
deepspeed --num_gpus 8 src/start_deepspeed.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 150 \
        --seq_len 120 \
        --num_experts 128 \
        --world_size 8 \
        --capacity_factor 50.0 \
        --router_skew 0.5 \
        --path "outputs/deepspeed"