# dataset=constant
# num_samples=200000

# cd ..
# python3 src/start_fastmoe.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --batch_size 1000 \
#         --seq_len 120 \
#         --world_size 8 \
#         --num_experts 128 \
#         --random_router_skew True \
#         --path "outputs/fastmoe/run"

dataset="random"
num_samples=25600
seq_len=512
batch_size=64
world_size=8
num_experts=128

cd ..
python3 src/start_fastmoe.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --num_experts 128 \
        --world_size $world_size \
        --pa "outputs/fastmoe/run"