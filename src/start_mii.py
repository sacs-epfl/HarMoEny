from mii import pipeline
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time
import os

from flexible_dataset import FlexibleDataset

num_iters = 10
seq_len = 512
batch_size = 128
num_gpus = 8
output_dir = "../outputs/"

pipe = pipeline("mistralai/Mixtral-8x7B-Instruct-v0.1") # max_length = seq_len + 1 (+1 for the generated)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/cache")

dataset = FlexibleDataset("random", tokenizer, None, seq_len=512, model_name="mixtral", num_samples=num_iters*batch_size*num_gpus, return_decoded=True)

loader = DataLoader(
    dataset, 
    batch_size=batch_size,
)

times = []
for batch in loader:
    start = time.time()
    output = pipe(batch, max_new_tokens=1, top_k=1)
    end = time.time()
    times.append(end-start)

with open(os.path.join(output_dir, "mii_deepspeed.csv"), "w") as f:
    for i, time in enumerate(times):
        f.write(f"{i},{time}\n")