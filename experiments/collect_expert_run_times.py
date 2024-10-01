import sys

from transformers import AutoModel
import torch
import random
import pandas as pd
from tqdm import tqdm

NUM_ITERS = 100000
AVG_OVER_NUM_ITERS = 3

model = AutoModel.from_pretrained("google/switch-base-8", cache_dir="/cache")
experts = model.encoder.block[1].layer[1].mlp.experts.to("cuda:0")

def get_expert(expert_idx):
    return experts[f"expert_{expert_idx}"]

dim = model.config.d_model

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def create_num_tokens_tensor(d):
    return torch.rand((d, dim), device="cuda:0")

# Assumes that you have len(list_num_tokens) number of experts
def get_average_time_for_num_tokens(list_num_tokens: [int]):
    # Warmup
    for i, num_tokens in enumerate(list_num_tokens):
        if num_tokens != 0:
            get_expert(i).forward(create_num_tokens_tensor(num_tokens))

    _sum = 0
    for _ in range(AVG_OVER_NUM_ITERS):
        tokens = [create_num_tokens_tensor(num_tokens) if num_tokens != 0 else None for num_tokens in list_num_tokens]
        torch.cuda.synchronize()
        start.record()
        for i, token in enumerate(tokens):
            if token is not None:
                get_expert(i).forward(token)
        end.record()
        torch.cuda.synchronize()
        _sum += start.elapsed_time(end)
    
    return _sum / AVG_OVER_NUM_ITERS

# print(get_average_time_for_num_tokens([10,0,0,50,25,0,0,0]))
# print(get_average_time_for_num_tokens([0,50,10,0,0,0,25,0]))

results = []
for _ in tqdm(range(NUM_ITERS)):
    num_experts = random.randint(0, len(experts))
    num_tokens = [0] * len(experts)
    for i in range(num_experts):
        num_tokens[i] = random.randint(0, 10000)
    results.append({
        "work": num_tokens,
        "time": get_average_time_for_num_tokens(num_tokens)
    })

df = pd.DataFrame(results)
df.to_csv("multi_expert_time_vs_tokens.csv")
