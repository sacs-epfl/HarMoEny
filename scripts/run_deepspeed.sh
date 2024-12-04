dataset=constant
num_samples=200000

cd ..
# deepspeed --num_gpus 8 src/start_deepspeed.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --batch_size 250 \
#         --seq_len 120 \
#         --num_experts 128 \
#         --world_size 8 \
#         --capacity_factor 64.0 \
#         --random_router_skew True \
#         --path "outputs/deepspeed/run"

deepspeed --num_gpus 8 src/start_deepspeed.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 10 \
        --seq_len 120 \
        --num_experts 128 \
        --world_size 8 \
        --router_skew 0.1 \
        --path "outputs/deepspeed/run"


        #         --random_router_skew True \
        #         --capacity_factor 64.0 \

