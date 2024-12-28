# https://deepspeed-mii.readthedocs.io/en/latest/pipeline.html#miipipeline
# https://deepspeed-mii.readthedocs.io/en/latest/config.html#mii.config.GenerateParamsConfig

from mii import pipeline
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
import time
import os

from flexible_dataset import FlexibleDataset

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

num_iters = 10
seq_len = 512
batch_size = 16
output_dir = "../outputs/"

pipe = pipeline("mistralai/Mixtral-8x7B-Instruct-v0.1") # max_length = seq_len + 1 (+1 for the generated)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/cache")

dataset = FlexibleDataset("random", tokenizer, None, seq_len=seq_len, model_name="mixtral", num_samples=num_iters*batch_size*world_size, return_decoded=True)
sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size, 
    rank=local_rank, 
    shuffle=True, 
    seed=49
)
loader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=batch_size,
)

try:
    times = []
    for batch in loader:
        start = time.time()
        output = pipe(batch, max_new_tokens=1, min_new_tokens=1, ignore_eos=True, top_k=1, do_sample=False, return_full_text=True, top_p=0.9) # do sample to False is eq to temperature = 0 but cannot have temp == 0
        end = time.time()
        times.append(end-start)

    with open(os.path.join(output_dir, "mii_deepspeed.csv"), "w") as f:
        for i, time in enumerate(times):
            f.write(f"{i},{time}\n")
finally:
    dist.destroy_process_group()