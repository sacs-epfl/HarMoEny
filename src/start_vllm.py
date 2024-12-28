# https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
# https://docs.vllm.ai/en/stable/dev/sampling_params.html

from vllm import LLM, SamplingParams
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time
import os

from flexible_dataset import FlexibleDataset

num_iters = 10
seq_len = 512
batch_size = 16
num_gpus = 8
output_dir = "../outputs/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/cache")
dataset = FlexibleDataset("random", tokenizer, None, seq_len=seq_len, model_name="mixtral", num_samples=num_iters*batch_size*num_gpus, return_decoded=True)

loader = DataLoader(
    dataset,
    batch_size=batch_size*num_gpus,
)

llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", task="generate", tensor_parallel_size=num_gpus, trust_remote_code=True)
sampling_params = SamplingParams(n=1, max_tokens=1, min_tokens=1, ignore_eos=True, top_k=1, temperature=0, detokenize=True, top_p=0.9)

try:
    times = []
    for batch in loader:
        start = time.time()
        output = llm.generate(batch, sampling_params=sampling_params)
        end = time.time()
        times.append(end-start)

    with open(os.path.join(output_dir, "vllm.csv"), "w") as f:
        for i, time in enumerate(times):
            f.write(f"{i},{time}\n")
finally:
    dist.destroy_process_group()

