# Point of this script is to find a closed form to estimate the 
# computation cost. This is helpful to then allocate resources
# on the fly to reduce the iteration time. 

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.wi = nn.Linear(768, 2048, bias=False)
        self.wo = nn.Linear(2048, 768, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


num_tokens = np.logspace(1, 20, num=1000, base=2).astype(int)
num_experts = [1, 2, 4, 8, 16]

results = []

NUM_ITERS = 4

torch.cuda.cudart().cudaProfilerStart()
for i in num_experts:
    experts = [SwitchTransformersDenseActDense() for _ in range(i)]
    for idx, expert in enumerate(experts):
        experts[idx] = expert.to("cuda:0")
    torch.cuda.synchronize()

    for j in num_tokens:
        if j < i:
            continue

        first = True
        for _iter in range(NUM_ITERS):
            torch.cuda.nvtx.range_push(f"exp_{i}_tokens_{j}_iter_{_iter}")
            num_toks_per_expert = j // i
            _input = torch.rand((i, num_toks_per_expert, 768), device="cuda:0")
            start_event.record()
            for idx, expert in enumerate(experts):
                expert.forward(_input[idx])
            end_event.record()
            torch.cuda.synchronize()
            if first:
                first = False
            else:
                results.append({
                    "num_experts": i,
                    "tot_num_tokens": i * num_toks_per_expert,
                    "time": start_event.elapsed_time(end_event), 
                })
            torch.cuda.nvtx.range_pop()

df = pd.DataFrame(results)
df.to_csv("exp2_results.csv", index=False)

print("Results saved to exp2_results.csv")
torch.cuda.cudart().cudaProfilerStop()
