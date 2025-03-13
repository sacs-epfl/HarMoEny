datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=100000
seq_len=1024
world_size=6
expert_cache_size=10
num_experts=128
expert_fetching_strategy="async-cpu"
eq_tokens=128
warmup_len=3
batch_size=20

cd ..
python3 src/start_harmony.py \
	--dataset bookcorpus \
	--num_samples $num_samples \
	--batch_size $batch_size \
	--seq_len $seq_len \
	--model_name "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
	--model_dtype "auto" \
	--loader transformers \
	--d_model 2048 \
	--num_experts 60 \
	--type_moe_parent Qwen2MoeDecoderLayer \
	--type_moe Qwen2MoeSparseMoeBlock \
	--router_tensor_path gate \
	--name_experts experts \
	--scheduling_policy harmony \
	--expert_cache_size $expert_cache_size \
	--world_size $world_size \
	--eq_tokens $eq_tokens \
	--expert_fetching_strategy $expert_fetching_strategy \
	--warmup_rounds $warmup_len \
	--pa "outputs/imbalance-qwen/$datetime"
