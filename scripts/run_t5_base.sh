dataset=bookcorpus
num_samples=1250

cd ..
python3 src/start_t5_base.py \
        --dataset $dataset \
        --num_samples $num_samples \
        --batch_size 250 \
        --seq_len 60 \
        --path "outputs/dense"