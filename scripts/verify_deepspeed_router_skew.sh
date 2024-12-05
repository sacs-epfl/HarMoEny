dataset=constant
num_samples=2000

cd ..
deepspeed --num_gpus 8 src/start_deepspeed.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 10 \
        --seq_len 120 \
        --num_experts 128 \
        --capacity_factor 128.0 \
        --world_size 8 \
        --enable_router_skew True \
        --router_skew 0.5 \
        --path "outputs/deepspeed_router_skew_check/yes"

deepspeed --num_gpus 8 src/start_deepspeed.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 10 \
        --seq_len 120 \
        --num_experts 128 \
        --capacity_factor 128.0 \
         --enable_router_skew True \
        --router_skew 0.0 \
        --world_size 8 \
        --path "outputs/deepspeed_router_skew_check/no"
