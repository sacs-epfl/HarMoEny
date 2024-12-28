num_samples=640
seq_len=512
batch_size=16
world_size=4

cd ..
python3 src/start_harmony.py \
        --dataset "random" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
        --scheduling_policy "harmony" \
        --expert_cache_size 2 \
        --world_size $world_size \
        --eq_tokens 1024 \
        --disable_async_fetch True \
        --pa "outputs/harmony/run"