from transformers import AutoModel
import torch
import pandas as pd 

AVG_OVER_NUM_ITERS = 3

model = AutoModel.from_pretrained("google/switch-base-8", cache_dir="/cache")
expert = model.encoder.block[1].layer[1].mlp.experts.expert_0.to("cuda:0")

dim = model.config.d_model

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def create_num_tokens_tensor(d):
    return torch.rand((d, dim), device="cuda:0")

def feed_value_to_expert_timed(tokens, expert):
    torch.cuda.synchronize()
    start.record()
    expert(tokens)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


results = []

for num_toks in [0,1,2,3,4,5]+list(range(10, 10000, 10)):
    # Warmup
    feed_value_to_expert_timed(create_num_tokens_tensor(num_toks), expert)

    _sum = 0
    for _ in range(AVG_OVER_NUM_ITERS):
        _sum += feed_value_to_expert_timed(create_num_tokens_tensor(num_toks), expert)
    results.append({
        "num_toks": num_toks,
        "time": _sum / AVG_OVER_NUM_ITERS
    })

df = pd.DataFrame(results)
df.to_csv("single_expert_time_vs_tokens.csv")
