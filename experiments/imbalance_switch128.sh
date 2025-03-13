datetime=$(date +"%Y-%m-%d_%H-%M")

num_samples=100000
seq_len=1024
world_size=8
expert_cache_size=16
num_experts=128
expert_fetching_strategy="async-cpu"
eq_tokens=128
warmup_len=3
batch_size=50

cd ..
python3 src/start_harmony.py \
	--dataset bookcorpus \
	--num_samples $num_samples \
	--batch_size $batch_size \
	--seq_len $seq_len \
	--model_name "google/switch-base-$num_experts" \
	--model_dtype "float" \
	--num_experts $num_experts \
	--scheduling_policy harmony \
	--expert_cache_size $expert_cache_size \
	--world_size $world_size \
	--eq_tokens $eq_tokens \
	--expert_fetching_strategy $expert_fetching_strategy \
	--warmup_rounds $warmup_len \
	--pa "outputs/imbalance-switch8/$datetime"
