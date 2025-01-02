datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=3840
# ONLINE (latency)
# seq_len=512
# batch_size=16
# OFFLINE (throughput)
seq_len=1024
batch_size=48
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
enable_skew=True
skew=90
num_experts_skewed=5
enable_random=False
enable_uniform=False
eq_tokens=2048
warmup_len=3

cd ..
# python3 src/start_harmony.py \
#         --dataset "bookcorpus" \
#         --num_samples $num_samples \
#         --batch_size $batch_size \
#         --seq_len $seq_len \
#         --model_name "google/switch-base-$num_experts" \
#         --num_experts $num_experts \
#         --scheduling_policy "deepspeed" \
#         --expert_cache_size $expert_cache_size \
#         --world_size $world_size \
#         --warmup_rounds $warmup_len \
#         --enable_router_skew $enable_skew \
#         --enable_router_random $enable_random \
#         --enable_router_uniform $enable_uniform \
#         --router_skew $skew \
#         --router_num_experts_skewed $num_experts_skewed \
#         --pa "outputs/timetime/$datetime/harmony-no-sched"

# python3 src/start_harmony.py \
#         --dataset "bookcorpus" \
#         --num_samples $num_samples \
#         --batch_size $batch_size \
#         --seq_len $seq_len \
#         --model_name "google/switch-base-$num_experts" \
#         --num_experts $num_experts \
#         --scheduling_policy "harmony" \
#         --expert_cache_size $expert_cache_size \
#         --world_size $world_size \
#         --eq_tokens $eq_tokens \
#         --expert_fetching_strategy "async-cpu" \
#         --warmup_rounds $warmup_len \
#         --enable_router_skew $enable_skew \
#         --enable_router_random $enable_random \
#         --enable_router_uniform $enable_uniform \
#         --router_skew $skew \
#         --router_num_experts_skewed $num_experts_skewed \
#         --pa "outputs/timetime/$datetime/harmony-async"


python3 src/start_harmony.py \
        --dataset "bookcorpus" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "google/switch-base-$num_experts" \
        --num_experts $num_experts \
        --scheduling_policy "harmony" \
        --expert_cache_size $expert_cache_size \
        --world_size $world_size \
        --eq_tokens $eq_tokens \
        --expert_fetching_strategy "async-cpu" \
        --warmup_rounds $warmup_len \
        --enable_router_skew $enable_skew \
        --enable_router_random $enable_random \
        --enable_router_uniform $enable_uniform \
        --router_skew $skew \
        --router_num_experts_skewed $num_experts_skewed \
        --pa "outputs/timetime/$datetime/harmony-async"

# torchrun \
#         --standalone \
#         --nnodes 1 \
#         --nproc-per-node $world_size \
#         src/start_harmony.py \
#                 --dataset "bookcorpus" \
#                 --num_samples $num_samples \
#                 --batch_size $batch_size \
#                 --seq_len $seq_len \
#                 --model_name "google/switch-base-$num_experts" \
#                 --num_experts $num_experts \
#                 --scheduling_policy "harmony" \
#                 --expert_cache_size $expert_cache_size \
#                 --world_size $world_size \
#                 --eq_tokens $eq_tokens \
#                 --expert_fetching_strategy "async-cpu" \
#                 --warmup_rounds $warmup_len \
#                 --enable_router_skew $enable_skew \
#                 --enable_router_random $enable_random \
#                 --enable_router_uniform $enable_uniform \
#                 --router_skew $skew \
#                 --router_num_experts_skewed $num_experts_skewed \
#                 --pa "outputs/timetime/$datetime/harmony-async"

# python3 src/start_harmony.py \
#         --dataset "bookcorpus" \
#         --num_samples $num_samples \
#         --batch_size $batch_size \
#         --seq_len $seq_len \
#         --model_name "google/switch-base-$num_experts" \
#         --num_experts $num_experts \
#         --scheduling_policy "harmony" \
#         --expert_cache_size $expert_cache_size \
#         --world_size $world_size \
#         --eq_tokens $eq_tokens \
#         --expert_fetching_strategy "sync-cpu" \
#         --warmup_rounds $warmup_len \
#         --enable_router_skew $enable_skew \
#         --enable_router_random $enable_random \
#         --enable_router_uniform $enable_uniform \
#         --router_skew $skew \
#         --router_num_experts_skewed $num_experts_skewed \
#         --pa "outputs/timetime/$datetime/harmony-sync"



# seq_len=12
# batch_size=1
# num_samples=1

# MIXTRAL
# python3 src/start_harmony.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --seq_len $seq_len \
#         --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
#         --num_experts 8 \
#         --scheduling_policy "harmony" \
#         --expert_cache_size 2 \
#         --world_size 4 \
#         --batch_size $batch_size \
#         --enable_router_skew True \
#         --router_skew 0.5 \
#         --router_num_experts_skew 2 \
#         --type_moe_parent "MixtralDecoderLayer" \
#         --type_moe "MixtralSparseMoeBlock" \
#         --name_router "gate" \
#         --name_experts "experts" \
#         --dynamic_components "w1" "w2" "w3" \
#         --d_model 4096 \
#         --eq_tokens 1024 \
#         --path "outputs/timetime/mixtral_multi/harmony_router_skew50-2"

# python3 src/start_fastmoe.py \
#         --dataset $dataset \
#         --num_samples $num_samples \
#         --seq_len $seq_len \
#         --world_size 8 \
#         --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
#         --num_experts 8 \
#         --batch_size $batch_size \
#         --enable_router_skew True \
#         --router_skew 0.5 \
#         --router_num_experts_skew 2 \
#         --pa "outputs/timetime/mixtral_multi/fastmoe_router_skew50"
