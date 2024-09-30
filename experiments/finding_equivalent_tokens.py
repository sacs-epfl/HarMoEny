# Make sure to use the standard untampered transformers implementation
from transformers import AutoModel
import torch
import sys

# FILE PARAMETERS
AVG_OVER_NUM_ITERS = 5
WARMUP_SIZE = 144


model = AutoModel.from_pretrained("google/switch-base-8", cache_dir="/cache")

# Just get the very first expert
expert = model.encoder.block[1].layer[1].mlp.experts.expert_0.to("cuda:0")

dim = model.config.d_model

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

results = []

def create_num_tokens_tensor(d):
    return torch.rand((d, dim), device="cuda:0")

def feed_value_to_expert_timed(tokens, expert):
    torch.cuda.synchronize()
    start.record()
    expert(tokens)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def get_average_time_for_num_tokens(num_tokens):
    # Warmup
    feed_value_to_expert_timed(create_num_tokens_tensor(num_tokens), expert)
    
    _sum = 0
    for _ in range(AVG_OVER_NUM_ITERS):
        run = feed_value_to_expert_timed(create_num_tokens_tensor(num_tokens), expert)
        # print(run)
        _sum += run
    return _sum / AVG_OVER_NUM_ITERS

zero = get_average_time_for_num_tokens(0)

# Need to find num tokens that achieve 2*zero
targets = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
results = []
for target in targets:
    val = get_average_time_for_num_tokens(target)
    results.append(val)

# Then we grab the largest bound that is out of bounds while the previous inside
bound = 1
for i in range(len(results)-1, 0, -1):
    if results[i-1] <= 2*zero:
        bound = i
        break

# Let us find the exact eq token now
eq_tokens = targets[bound-1]
for num_toks in range(targets[bound], targets[bound-1], -1):
    if get_average_time_for_num_tokens(num_toks-1) <= 2*zero:
        eq_tokens = num_toks 
        break

print(eq_tokens)

