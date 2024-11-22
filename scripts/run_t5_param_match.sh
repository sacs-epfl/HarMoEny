dataset=bookcorpus
num_samples=1250

cd ..
python3 src/start_t5_param_amt_match.py \
        --num_experts 32 \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 250 \
        --seq_len 60 \
        --path "outputs/dense_param_match"