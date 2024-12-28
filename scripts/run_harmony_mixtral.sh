num_samples=1280
seq_len=512
batch_size=16
world_size=8

cd ..
python3 src/start_harmony.py \
        --dataset "random" \
        --num_samples $num_samples \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
        --num_experts 8 \
        --d_model 4096 \
        --type_moe_parent MixtralDecoderLayer \
        --type_moe MixtralSparseMoeBlock \
        --name_router gate \
        --name_experts experts \
        --dynamic_components w1 w2 w3 \
        --scheduling_policy "harmony" \
        --expert_cache_size 1 \
        --world_size $world_size \
        --eq_tokens 1024 \
        --disable_async_fetch True \
        --pa "outputs/harmony"